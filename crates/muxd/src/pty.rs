//! PTY spawn and async IO. Fork of ix-console's pty.rs / pty/exec.rs
//! core: openpty, fork, setsid + `TIOCSCTTY`, dup2, execvp; the master fd
//! becomes a tokio `AsyncFd` (no `spawn_blocking` reads).

use std::ffi::CString;
use std::os::fd::{AsRawFd, OwnedFd};
use std::path::PathBuf;

use anyhow::{Context, Result};
use nix::pty::Winsize;
use nix::unistd::{ForkResult, Pid, User};
use tokio::io::unix::AsyncFd;

pub struct Pty {
    pub master: AsyncFd<OwnedFd>,
    pub child: Pid,
}

pub struct Spawn<'a> {
    /// argv; empty means the user's login shell.
    pub command: &'a [String],
    pub cwd: Option<&'a str>,
    pub term: Option<&'a str>,
    pub cols: u16,
    pub rows: u16,
}

/// execvp's PATH search, done before fork: the child execs with execve
/// to stay async-signal-safe, so any lookup must happen here. A name
/// that resolves nowhere is returned as-is (execve fails, child exits
/// 127).
fn resolve_in_path(path: CString) -> CString {
    use std::os::unix::ffi::OsStringExt;
    use std::os::unix::fs::PermissionsExt;
    let name = path.to_string_lossy();
    if name.contains('/') {
        return path;
    }
    if let Some(dirs) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&dirs) {
            let candidate = dir.join(&*name);
            let executable = candidate
                .metadata()
                .is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0);
            if executable {
                if let Ok(resolved) = CString::new(candidate.into_os_string().into_vec()) {
                    return resolved;
                }
            }
        }
    }
    path
}

fn winsize(cols: u16, rows: u16) -> Winsize {
    Winsize {
        ws_row: rows.max(1),
        ws_col: cols.max(1),
        ws_xpixel: 0,
        ws_ypixel: 0,
    }
}

/// A daemon under a service manager has no $SHELL or $HOME: ask passwd,
/// like login does. A service account's nologin surfaces verbatim rather
/// than being masked, because the daemon must run as the human whose
/// shells it spawns.
fn passwd() -> Option<User> {
    User::from_uid(nix::unistd::Uid::current()).ok().flatten()
}

fn env_or<F: FnOnce(&User) -> PathBuf>(var: &str, field: F, fallback: &str) -> String {
    if let Ok(value) = std::env::var(var) {
        if !value.is_empty() {
            return value;
        }
    }
    passwd()
        .map(|user| field(&user))
        .and_then(|path| path.into_os_string().into_string().ok())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| fallback.to_string())
}

fn user_shell() -> String {
    env_or("SHELL", |user| user.shell.clone(), "/bin/zsh")
}

fn user_home() -> String {
    env_or("HOME", |user| user.dir.clone(), "/")
}

/// The tokio reactor requires it, and an inherited fd (migrate.rs) may
/// arrive without it.
pub fn set_nonblocking(fd: &OwnedFd) -> Result<()> {
    let flags = nix::fcntl::fcntl(fd.as_raw_fd(), nix::fcntl::FcntlArg::F_GETFL)?;
    let mut oflags = nix::fcntl::OFlag::from_bits_truncate(flags);
    oflags.insert(nix::fcntl::OFlag::O_NONBLOCK);
    nix::fcntl::fcntl(fd.as_raw_fd(), nix::fcntl::FcntlArg::F_SETFL(oflags))?;
    Ok(())
}

/// A pty's window size in cells.
pub struct WindowSize {
    pub cols: u16,
    pub rows: u16,
}

/// Current size of the pty, falling back to the protocol default when the
/// ioctl fails or reports an unset size.
pub fn window_size(master: &AsyncFd<OwnedFd>) -> WindowSize {
    let mut ws = winsize(0, 0);
    let ok =
        unsafe { libc::ioctl(master.get_ref().as_raw_fd(), libc::TIOCGWINSZ, &raw mut ws) } == 0;
    if ok && ws.ws_col > 0 && ws.ws_row > 0 {
        WindowSize {
            cols: ws.ws_col,
            rows: ws.ws_row,
        }
    } else {
        WindowSize {
            cols: mux_proto::peer::DEFAULT_COLS,
            rows: mux_proto::peer::DEFAULT_ROWS,
        }
    }
}

pub fn spawn(params: &Spawn) -> Result<Pty> {
    let pty =
        nix::pty::openpty(Some(&winsize(params.cols, params.rows)), None).context("openpty")?;

    // Everything the child needs, allocated before fork: only
    // async-signal-safe calls are allowed after.
    let (exec_path, exec_argv): (CString, Vec<CString>) = if params.command.is_empty() {
        // Login shell convention: argv[0] = "-zsh".
        let shell = user_shell();
        let name = shell.rsplit('/').next().unwrap_or(&shell);
        (
            CString::new(shell.as_str())?,
            vec![CString::new(format!("-{name}"))?],
        )
    } else {
        let argv: Vec<CString> = params
            .command
            .iter()
            .map(|a| CString::new(a.as_str()))
            .collect::<Result<_, _>>()?;
        (argv[0].clone(), argv)
    };
    // No explicit cwd: start at home, like login does. The daemon's own
    // cwd (a service manager's "/") is never a useful shell start dir.
    let cwd = match params.cwd {
        Some(dir) => Some(CString::new(dir)?),
        None => CString::new(user_home()).ok(),
    };
    let term = CString::new(format!("TERM={}", params.term.unwrap_or("xterm-ghostty")))?;
    // Panes are always rendered by a truecolor terminal; advertise it the
    // way the renderer itself would (the daemon's own environment has no
    // COLORTERM, and the child inherits it otherwise).
    let colorterm = CString::new("COLORTERM=truecolor")?;
    // ncurses/readline resolve TERM=xterm-ghostty via TERMINFO, which is
    // absent from every system terminfo db (it only ships inside Ghostty.app
    // and Mux.app's own bundles). Without it the pane shell can't find kbs
    // (key_backspace = \177) and erase rendering breaks: backspace prints a
    // space instead of deleting. GHOSTTY_RESOURCES_DIR (set by main.swift,
    // e.g. .../Resources/ghostty) is the one thing the daemon's env is
    // guaranteed to carry; terminfo lives at the sibling path
    // .../Resources/terminfo (same layout make-app.sh assembles).
    let terminfo = std::env::var_os("GHOSTTY_RESOURCES_DIR").and_then(|dir| {
        use std::os::unix::ffi::OsStringExt;
        let sibling = PathBuf::from(dir).parent()?.join("terminfo");
        let mut bytes = b"TERMINFO=".to_vec();
        bytes.extend(sibling.into_os_string().into_vec());
        CString::new(bytes).ok()
    });

    // Everything below is built BEFORE fork. The daemon runs a
    // multithreaded runtime, so the child may only make
    // async-signal-safe calls: no allocation (another thread may hold
    // the malloc lock at fork time), which rules out putenv, execvp's
    // PATH search, and building pointer tables after the fork.
    let exec_path = resolve_in_path(exec_path);
    let mut env: Vec<CString> = std::env::vars_os()
        .filter(|(k, _)| match k.to_str() {
            // TERM/COLORTERM/TERMINFO are re-added below. CLAUDE*/AI_AGENT
            // are per-session markers of whatever agent happened to start
            // the daemon; leaking them makes every pane shell look like a
            // nested agent session (e.g. claude disables transcript
            // saving under CLAUDE_CODE_CHILD_SESSION).
            Some(k) => {
                !matches!(k, "TERM" | "COLORTERM" | "TERMINFO" | "AI_AGENT")
                    && !k.starts_with("CLAUDE")
            }
            None => true,
        })
        .filter_map(|(k, v)| {
            use std::os::unix::ffi::OsStringExt;
            let mut bytes = k.into_vec();
            bytes.push(b'=');
            bytes.extend(v.into_vec());
            CString::new(bytes).ok()
        })
        .collect();
    env.push(term);
    env.push(colorterm);
    if let Some(terminfo) = terminfo {
        env.push(terminfo);
    }
    let envp: Vec<*const libc::c_char> = env
        .iter()
        .map(|e| e.as_ptr())
        .chain([std::ptr::null()])
        .collect();
    let argv_ptrs: Vec<*const libc::c_char> = exec_argv
        .iter()
        .map(|a| a.as_ptr())
        .chain([std::ptr::null()])
        .collect();

    match unsafe { nix::unistd::fork() }.context("fork")? {
        ForkResult::Parent { child } => {
            drop(pty.slave);
            let master = pty.master;
            set_nonblocking(&master)?;
            Ok(Pty {
                master: AsyncFd::new(master).context("AsyncFd")?,
                child,
            })
        }
        ForkResult::Child => {
            // Async-signal-safe zone. Any failure: _exit(127).
            let slave = pty.slave.as_raw_fd();
            unsafe {
                if libc::setsid() < 0 {
                    libc::_exit(127);
                }
                if libc::ioctl(slave, u64::from(libc::TIOCSCTTY), 0) < 0 {
                    libc::_exit(127);
                }
                libc::dup2(slave, 0);
                libc::dup2(slave, 1);
                libc::dup2(slave, 2);
                if slave > 2 {
                    libc::close(slave);
                }
                libc::close(pty.master.as_raw_fd());
                if let Some(dir) = &cwd {
                    // Best-effort; a stale cwd should not kill the shell.
                    let _ = libc::chdir(dir.as_ptr());
                }
                libc::execve(exec_path.as_ptr(), argv_ptrs.as_ptr(), envp.as_ptr());
                libc::_exit(127);
            }
        }
    }
}

/// Write all of `data` to the PTY, waiting for writability.
pub async fn write_all(master: &AsyncFd<OwnedFd>, data: &[u8]) -> Result<()> {
    let mut written = 0;
    while written < data.len() {
        let mut guard = master.writable().await?;
        match guard.try_io(|fd| {
            nix::unistd::write(fd.get_ref(), &data[written..])
                .map_err(|e| std::io::Error::from_raw_os_error(e as i32))
        }) {
            Ok(Ok(n)) => written += n,
            Ok(Err(e)) if e.kind() == std::io::ErrorKind::WouldBlock => {}
            Ok(Err(e)) => return Err(e.into()),
            Err(_would_block) => {}
        }
    }
    Ok(())
}

pub fn resize(master: &AsyncFd<OwnedFd>, cols: u16, rows: u16) -> Result<()> {
    let ws = winsize(cols, rows);
    let res = unsafe { libc::ioctl(master.get_ref().as_raw_fd(), libc::TIOCSWINSZ, &ws) };
    if res < 0 {
        return Err(std::io::Error::last_os_error()).context("TIOCSWINSZ");
    }
    Ok(())
}
