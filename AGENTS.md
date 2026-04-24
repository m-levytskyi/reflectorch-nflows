# Personal AGENTS.md

## Core Operating Principles

### 1. Think Before Coding
- Do not assume missing details.
- State assumptions explicitly before implementing when they affect design or behavior.
- When requirements are ambiguous, present the plausible interpretations and choose only when justified by the repository or request context.
- Push back when a simpler or safer approach is better than the requested shape.
- Stop and surface confusion instead of coding through uncertainty.
- Check whether the logic already exists before writing new code.

### 2. Simplicity First
- Solve the requested problem with the minimum code necessary.
- Do not add features, flexibility, configurability, or abstractions that were not requested.
- No speculative generalization.
- No abstractions for single-use code.
- Prefer simple single-purpose functions over multi-mode behavior.
- If a solution can be made materially smaller and clearer without losing correctness, simplify it.

### 3. Surgical Changes
- Keep changes minimal and directly related to the current request.
- Touch only the code required for the task.
- Do not refactor adjacent code, comments, formatting, or architecture unless the task requires it.
- Match the existing repository style even when it differs from my personal preference.
- Do not revert unrelated changes.
- Remove only the dead code, imports, variables, and functions made unused by the current change.
- If unrelated issues are noticed, mention them separately instead of fixing them opportunistically.

### 4. Goal-Driven Execution
- Define the success criteria before making non-trivial changes.
- For multi-step work, write a brief plan where each step has a verification method.
- Prefer verifiable outcomes over imperative descriptions of work.
- When fixing bugs, reproduce the issue first when practical, then verify the fix.
- When changing behavior, run the smallest meaningful validation that proves the requested outcome.
- Keep iterating until the success criteria are verified or a real blocker is identified explicitly.

## Code Style

- Comments in English only
- Prefer functional programming over OOP
- Use OOP classes only for connectors and interfaces to external systems
- Write pure functions - only modify return values, never input parameters or global state
- Follow DRY, KISS, and YAGNI principles
- Use strict typing everywhere - function returns, variables, collections
- Avoid untyped variables and generic types
- Never use default parameter values - make all parameters explicit
- Create proper type definitions for complex data structures
- All imports at the top of the file
- Write simple single-purpose functions - no multi-mode behavior, no flag parameters that switch logic
- Prefer explicit data flow over hidden state
- Prefer composition over inheritance
- Name things so the code explains itself before adding documentation

## Error Handling

- Always raise errors explicitly, never silently ignore them
- Use specific error types that clearly indicate what went wrong
- Avoid catch-all exception handlers that hide the root cause
- Error messages should be clear and actionable
- No fallbacks unless I explicitly ask for them
- Fix root causes, not symptoms
- Do not add defensive handling for impossible or unsupported scenarios unless the request or existing system requires it
- External API or service calls: use retries with warnings, then raise the last error
- Error messages must include enough context to debug: request params, response body, status codes
- Logging should use structured fields instead of interpolating dynamic values into message strings

## Tooling and Dependencies

- Prefer modern package management files like `pyproject.toml` and `package.json`
- Install dependencies in project environments, not globally
- Add dependencies to project config files, not as one-off manual installs
- Read installed dependency source code when needed instead of guessing behavior
- Prefer existing project tools and patterns over introducing new ones
- Do not add new dependencies when the standard library or current stack already solves the problem cleanly

## Testing and Verification

- Respect the current repository testing strategy and existing test suite
- Do not add new unit tests by default
- When verification is needed, choose the lightest test or validation that proves the requested behavior in the real system
- Prefer integration, end-to-end, or smoke tests that validate real behavior
- Use unit tests only rarely, mainly for stable datasets or pure data transformations
- Never add unit tests just to increase coverage numbers
- Avoid mocks when real calls are practical
- It is usually better to spend a little money on real API or service calls than to maintain fragile mock-based coverage
- Add only the minimum test coverage needed for the requested change
- For bug fixes, prefer a reproducible failing check before the fix and a passing check after it
- For refactors, verify behavior before and after with existing tests or equivalent checks

## Codex Workflow

- Inspect the repository before editing
- Read active `AGENTS.md` files before making assumptions
- For non-trivial tasks, identify assumptions, success criteria, and the validation approach before editing
- Keep changes minimal and directly related to the current request
- Match the existing repository style even when it differs from my personal preference
- Do not revert unrelated changes
- Prefer `rg` for code search
- Use non-interactive commands with flags
- Always use non-interactive git diff: `git --no-pager diff` or `git diff | cat`
- Run relevant tests or validation commands after code changes when the project already defines them
- Do not claim success until the relevant validation has been run
- If validation cannot be run, state that explicitly and explain why

## Documentation

- Code is the primary documentation - use clear naming, types, and docstrings
- Keep documentation in docstrings of the functions or classes they describe, not in separate files
- Separate docs files only when a concept cannot be expressed clearly in code
- Never duplicate documentation across files
- Store knowledge as current state, not as a changelog of modifications
- Do not rewrite unrelated comments or docs while making functional changes
