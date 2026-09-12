"""The LanguageBackend interface.

The core pipeline (llm.py / base.py / personas) is target-language-agnostic: it
orchestrates the TDD methodology (oracle-first generation, diagnose -> fix,
revert-on-regression guardrail, --optimize persona loops) and delegates every
language-specific leaf to a LanguageBackend:

- naming / artifact contract (source, test, manifest, fence language, layout)
- generation and review prompts (full templates, owned per backend)
- validation of LLM markdown output
- compose (impl + oracle -> on-disk layout)
- formatting, linting, test execution
- the runnable-CLI step and per-backend reviewer guidance
- toolchain detection

Only `python` is wired in today; the registry in `marsha.backends` is ready for
more (issue #204; Rust is the first follow-on, #183).
"""


class LanguageBackend:
    """One target language's leaf operations. Subclasses implement every method."""

    id = ''
    aliases = ()
    code_fence_lang = ''

    # --- naming / contract ---------------------------------------------------

    def source_name(self, fn):
        """On-disk name of the implementation file for function name `fn`."""
        raise NotImplementedError

    def test_name(self, fn):
        """On-disk name of the test suite (oracle) file for function name `fn`."""
        raise NotImplementedError

    def manifest_name(self):
        """On-disk name of the dependency manifest, if the target has one."""
        raise NotImplementedError

    def artifact_contract(self, fn):
        """Ordered (kind, filename) pairs making up the full artifact set for
        function name `fn`. The single source of truth that the prompts describe
        and compose()/validate_markdown() enforce."""
        raise NotImplementedError

    def code_block(self, code):
        """Fence a snippet of source code for embedding in a prompt."""
        return f'```{self.code_fence_lang}\n{code}\n```'

    # --- generation / review prompts (full templates, per backend) -----------

    def spec_check_prompt(self):
        """System prompt for the spec sanity check (compilable? warnings/errors?)."""
        raise NotImplementedError

    def oracle_prompt(self, meta):
        """System prompt generating the oracle (the test suite) from the assignment."""
        raise NotImplementedError

    def impl_prompt(self, meta):
        """System prompt generating an implementation against the oracle."""
        raise NotImplementedError

    def diagnose_prompt(self):
        """System prompt deciding whether a test failure is the impl's or a test's fault."""
        raise NotImplementedError

    def fix_impl_prompt(self, meta):
        """System prompt fixing the implementation (the oracle is off-limits)."""
        raise NotImplementedError

    def correct_test_prompt(self, meta):
        """System prompt for the spec-anchored test correction (the oracle punt-back)."""
        raise NotImplementedError

    def lint_fix_prompt(self, filename):
        """System prompt fixing lint findings in a single file."""
        raise NotImplementedError

    # --- layout / validation --------------------------------------------------

    def compose(self, impl, oracle):
        """Compose an implementation doc and the canonical oracle doc into the
        on-disk layout: an ordered {path: content} mapping. The oracle is
        harness-owned and the impl is produced against it, so recomposing on
        every write keeps the impl path from touching the oracle in any layout."""
        raise NotImplementedError

    def validate_markdown(self, doc, kind, name):
        """Validate an LLM markdown response against the artifact contract.
        kind is 'oracle' (the test suite; name is the function name), 'impl'
        (implementation + optional manifest; name is the function name), or
        'unit' (a single file whose first header must be `name` verbatim)."""
        raise NotImplementedError

    # --- toolchain leaves -------------------------------------------------------

    def format_files(self, paths):
        """Format the given source files in place (e.g. autopep8)."""
        raise NotImplementedError

    def lint_files(self, paths):
        """Lint the given files and return {path: [finding, ...]}."""
        raise NotImplementedError

    async def run_tests(self, code, test, manifest, debug=False):
        """Build/install/run the test suite. Returns (passed, results): passed is
        None if the suite could not be run at all, False if it ran and failed,
        True if it passed."""
        raise NotImplementedError

    def make_executable(self, path):
        """Make a generated module a runnable CLI. Python appends the static
        reflection helper; other targets may use an LLM iteration instead."""
        raise NotImplementedError

    def persona_guidance(self):
        """Per-backend conventions injected into every reviewer persona run
        (language/project specifics the shared, language-agnostic reviewer
        bodies deliberately leave out)."""
        raise NotImplementedError

    def toolchain(self):
        """The interpreter/compiler executable for the target (raises if missing)."""
        raise NotImplementedError

    def toolchain_ok(self):
        """Whether the target toolchain is available (no raising)."""
        raise NotImplementedError
