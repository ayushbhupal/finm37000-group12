# Contributing

## Workflow

1. **Pick an issue** and assign yourself so nobody duplicates work.
2. **Branch from `main`**, named after the issue: `git checkout -b 7-backtest-engine`.
3. **Commit small, focused changes** with messages that say what changed.
4. **Run `pytest`** before pushing. New code should come with at least one test.
5. **Open a pull request** into `main`, put `Closes #<issue>` in the description,
   and request a review from both teammates.
6. **Review each other's PRs.** At least one approval before merging. The author
   merges after addressing comments.

If your issue depends on one that isn't merged yet, code against the function
signatures in `src/` and use simple fakes in your tests, then swap in the real
implementation once it lands.

## Conventions

- Library code goes in `src/`; notebooks in `notebooks/` should import from `src`
  rather than copy code.
- Parameters belong in `config.yaml`, not hard-coded.
- Never commit data files, `.env`, or API keys. They are git-ignored.
- Contract details (multiplier, tick size) come from `src/contracts.py`.

## Communication

Day-to-day coordination happens in the team group chat; decisions that affect
the code (strategy choice, roll rule, data source) should also be written down
in the relevant issue so they are on record.
