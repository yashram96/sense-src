# Sense documentation site

This is a [Mintlify](https://mintlify.com) project — `docs.json` plus MDX
pages. It's the polished, reader-facing documentation for the Sense
language; it complements (doesn't replace) [`../internal/`](../internal), which is
the internal, denser design log (`ROADMAP.md`, `LANGUAGE_v0.1–v0.3.md`).

## Preview locally

```bash
npm i -g mint
mint dev
```

Opens at `http://localhost:3000` by default.

> As of this writing, `npx mint@latest` intermittently fails to resolve one
> of its own internal dependency versions (`@mintlify/models`) — an
> upstream npm registry issue, not something wrong with this project. If
> `mint dev` fails to install, retry, or pin an explicit recent `mint`
> version (`npm i -g mint@<version>`).

## Deploy

Connect this repository in the [Mintlify dashboard](https://dashboard.mintlify.com)
and point it at this `docs/` directory, or use `mint deploy` if your plan
supports CLI deploys.

## Structure

```text
docs.json              Site config: theme, colors, navigation
index.mdx               Introduction
quickstart.mdx           Install + first program
installation.mdx         Detailed setup, project layout, publishing
philosophy/              Why Sense exists, its computational model, design
                         order, and explicit non-goals
language/                The deterministic core: values, scope, functions,
                         control flow, arrays, modules
ai-native/                model/ask/answer, session, agent, pause/resume
safety/                   action (prepare -> verify -> commit), policy
comparison.mdx            Sense vs. Python / the conventional AI stack
roadmap.mdx               Phase-by-phase status, kept in sync with
                         ../internal/ROADMAP.md
faq.mdx                   Common questions
reference/                Grammar, builtins, errors, CLI
architecture/             How the interpreter actually works internally
```

## Keeping this in sync

Every time a language feature ships or changes (a new keyword, a changed
grammar rule, a resolved "not built yet" item), the corresponding page(s)
here should be updated in the same change — the way `../internal/ROADMAP.md`
and `../internal/LANGUAGE_v0.*.md` already are. This site is meant to be
incrementally maintained, not written once and left to drift.
