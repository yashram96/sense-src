<p align="center">
  <img src="https://raw.githubusercontent.com/yashram96/sense-src/master/images/sense_file_icon.png" alt="" width="84">
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/yashram96/sense-src/master/images/sense_logo_white_transarent.png">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/yashram96/sense-src/master/images/sense_logo_black_transarent.png">
    <img src="https://raw.githubusercontent.com/yashram96/sense-src/master/images/sense_logo_black_transarent.png" alt="Sense" width="280">
  </picture>
</p>

<p align="center">
  A programming language for probabilistic and autonomous computation —
  where a model's answer, a staged side effect, a permission check, and a
  human approval gate are things the language itself understands, not
  conventions a library hopes you follow.
</p>

<p align="center">
  <a href="https://pypi.org/project/sense-lang/"><img src="https://img.shields.io/pypi/v/sense-lang.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/sense-lang/"><img src="https://img.shields.io/pypi/pyversions/sense-lang.svg" alt="Python versions"></a>
  <a href="https://github.com/yashram96/sense-src/blob/master/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT licensed"></a>
</p>

<p align="center">
  <a href="https://sensecode.dev">Website</a> ·
  <a href="https://docs.sensecode.dev">Documentation</a>
</p>

---

## Install

```bash
pip install sense-lang
```

Requires Python 3.10+. No other runtime dependency for the interpreter
itself.

Optional extras, only needed if you use the matching feature:

```bash
pip install sense-lang[anthropic]   # inference("anthropic", ...)
pip install sense-lang[openai]      # inference("openai", ...)
pip install sense-lang[mcp]         # connect_mcp(...)
```

## Quickstart

```sns
def fib(n) -> Int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

i = 0
while i < 10:
    print(fib(i))
    i = i + 1
```

```bash
sense run fib.sns
sense repl
```

Reasoning about something never talks to a vendor SDK directly — it asks
whichever model is currently delegated to, and reads back a structured
`Answer`, never a plain string:

```sns
model reasoning = inference("mock", "demo")   # offline, no API key needed
set delegation = reasoning

result = ask("what database should we use?")
print(result.value)
print(result.confidence)
```

A side effect that changes the outside world gets its own lifecycle —
calling it never runs the effect; only `.commit()` does:

```sns
irreversible action send_mail(to: String, body: String) requires email.send:
    print("sending to " + to)

policy: allow email.send

mail = send_mail("alice@example.com", "hi")   # nothing sent yet
mail.verify()
mail.commit()                                  # only now
```

Full walkthroughs, the language reference, and the safety model live at
**[docs.sensecode.dev](https://docs.sensecode.dev)**.

## What's in the box

- **`ask`/`Model`/`Answer`** — vendor-agnostic reasoning, with confidence
  and source structurally distinct from a plain value.
- **`action`** — prepare → verify → commit, enforced by the interpreter;
  a `reversible action` must declare how to undo itself.
- **`policy`/`requires`** — capability-gated permissions, scoped exactly
  like variables.
- **`agent`** — a real runtime entity with identity, state, and a genuine
  pause/resume lifecycle.
- **`tool`/`skill`/`connect_mcp`** — capability-checked functions a model
  can be handed to call on its own.
- **`memory`/`persistent memory`** — session-scoped and durable,
  versioned key/value storage.
- **`import python "module"`** — reach into any installed Python package
  directly; the interpreter is itself written in Python.

## Development

Contributing or building from source:

```bash
git clone <repo-url>
cd sense
python -m venv .venv
.venv/Scripts/python -m pip install -e .[dev]   # Windows
# source .venv/bin/activate && pip install -e .[dev]   # macOS/Linux

pytest -q
```

## License

MIT — see [LICENSE](https://github.com/yashram96/sense-src/blob/master/LICENSE).
