"""The LanguageBackend interface.

The core pipeline (llm.py / base.py / personas) is target-language-agnostic: it
orchestrates the TDD methodology (oracle-first generation, diagnose -> fix,
revert-on-regression guardrail, --optimize persona loops) and delegates every
language-specific leaf to a LanguageBackend:

- target version resolution (the --target-version CLI value, per-target default)
- naming / artifact contract (source, test, manifest, fence language, layout)
- generation and review prompts (full templates, owned per backend)
- validation of LLM markdown output
- compose (impl + oracle -> on-disk layout)
- formatting, linting, test execution
- the runnable-CLI step and per-backend reviewer guidance
- toolchain detection
- the fake-terminal tools: the language-agnostic set (web-search, view-web-page,
  calc) is defined once in `marsha.tools`; a backend layers its language-specific
  tools (package registry + installed-env introspection) on top of it.

Only `python` is wired in today; the registry in `marsha.backends` is ready for
more (issue #204; Rust is the first follow-on, #183).
"""

from marsha.meta import MarshaMeta
from marsha.tools import agnostic_tool_commands, ToolContext, ToolCommand


class LanguageBackend:
    """One target language's leaf operations. Subclasses implement every method."""

    id: str = ''
    aliases: tuple[str, ...] = ()
    code_fence_lang: str = ''
    target_version: str = ''

    # --- fake-terminal tools ----------------------------------------------------

    def tool_commands(self, ctx: ToolContext) -> dict[str, ToolCommand]:
        """The fake-terminal commands for this target language, before phase
        scoping: the language-agnostic set (web-search, view-web-page, calc),
        defined once in `marsha.tools`, plus this backend's language-specific
        tools. A concrete backend layers its registry and installed-env tools on
        top of `super().tool_commands(ctx)`, so adding a target only adds its
        registry/env tools, never re-implements the web/computation tools."""
        return dict(agnostic_tool_commands(ctx))

    def installed_env_usable(self, ctx: ToolContext) -> bool:
        """Whether this backend's installed-env tools are available for `ctx`
        (e.g. the candidate environment exists for the phase's working dir).
        False by default: only backends that provide installed-env tools
        (and where its environment exists) return True. The *mechanism* is
        backend-internal — this is the only installed-env hook the core calls."""
        return False

    def resolve_target_version(self, requested: str | None) -> str:
        """Validate a --target-version value for this target and return its normalized
        form; a requested value of None means 'use this target's default'. The resolved
        value lives on the backend instance as target_version (set in the constructor
        and rebound by the CLI when the flag is given), and the backend's prompts
        render it into the generated artifacts."""
        raise NotImplementedError

    # --- naming / contract ---------------------------------------------------

    def source_name(self, fn: str) -> str:
        """On-disk name of the implementation file for function name `fn`."""
        raise NotImplementedError

    def test_name(self, fn: str) -> str:
        """On-disk name of the test suite (oracle) file for function name `fn`."""
        raise NotImplementedError

    def manifest_name(self) -> str:
        """On-disk name of the dependency manifest, if the target has one."""
        raise NotImplementedError

    def artifact_contract(self, fn: str) -> list[tuple[str, str]]:
        """Ordered (kind, filename) pairs making up the full artifact set for
        function name `fn`. The single source of truth that the prompts describe
        and compose()/validate_markdown() enforce."""
        raise NotImplementedError

    def code_block(self, code: str) -> str:
        """Fence a snippet of source code for embedding in a prompt."""
        return f'```{self.code_fence_lang}\n{code}\n```'

    # --- generation / review prompts (full templates, per backend) -----------

    def oracle_prompt(self, meta: MarshaMeta) -> str:
        """System prompt generating the oracle (the test suite) from the assignment."""
        raise NotImplementedError

    def impl_prompt(self, meta: MarshaMeta) -> str:
        """System prompt generating an implementation against the oracle."""
        raise NotImplementedError

    def diagnose_prompt(self) -> str:
        """System prompt deciding whether a test failure is the impl's or a test's fault."""
        raise NotImplementedError

    def fix_impl_prompt(self, meta: MarshaMeta) -> str:
        """System prompt fixing the implementation (the oracle is off-limits)."""
        raise NotImplementedError

    def correct_test_prompt(self, meta: MarshaMeta) -> str:
        """System prompt for the spec-anchored test correction (the oracle punt-back)."""
        raise NotImplementedError

    def lint_fix_prompt(self, filename: str) -> str:
        """System prompt fixing lint findings in a single file."""
        raise NotImplementedError

    # --- layout / validation --------------------------------------------------

    def compose(self, impl: str, oracle: str) -> dict[str, str]:
        """Compose an implementation doc and the canonical oracle doc into the
        on-disk layout: an ordered {path: content} mapping. The oracle is
        harness-owned and the impl is produced against it, so recomposing on
        every write keeps the impl path from touching the oracle in any layout."""
        raise NotImplementedError

    def validate_markdown(self, doc: str, kind: str, name: str) -> bool:
        """Validate an LLM markdown response against the artifact contract.
        kind is 'oracle' (the test suite; name is the function name), 'impl'
        (implementation + optional manifest; name is the function name), or
        'unit' (a single file whose first header must be `name` verbatim)."""
        raise NotImplementedError

    # --- toolchain leaves -------------------------------------------------------

    def format_files(self, paths: list[str]) -> None:
        """Format the given source files in place (e.g. autopep8)."""
        raise NotImplementedError

    def lint_files(self, paths: list[str]) -> dict[str, list[str]]:
        """Lint the given files and return {path: [finding, ...]}."""
        raise NotImplementedError

    async def run_tests(self, code_file: str, test_file: str, manifest: str | None,
                        debug: bool = False) -> tuple[bool | None, str]:
        """Build/install/run the test suite. Returns (passed, results): passed is
        None if the suite could not be run at all, False if it ran and failed,
        True if it passed."""
        raise NotImplementedError

    def make_executable(self, path: str) -> None:
        """Make a generated module a runnable CLI. Python appends the static
        reflection helper; other targets may use an LLM iteration instead."""
        raise NotImplementedError

    def persona_guidance(self) -> str:
        """Per-backend conventions injected into every reviewer persona run
        (language/project specifics the shared, language-agnostic reviewer
        bodies deliberately leave out)."""
        raise NotImplementedError

    def toolchain(self) -> str:
        """The interpreter/compiler executable for the target (raises if missing)."""
        raise NotImplementedError

    def toolchain_ok(self) -> bool:
        """Whether the target toolchain is available (no raising)."""
        raise NotImplementedError
