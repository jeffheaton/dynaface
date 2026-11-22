# Releasing

A library release is three packages with one version:
- `dynaface` on PyPI;
- `Dynaface` and `Dynaface.Onnx` on NuGet.org.

They ship together and share one tag, `v<version>`. The release automation
follows the release convention in `~/.claude/CLAUDE.md`, which heaton-voxel
follows too. Each language takes two manual steps:
- A **build** tests the code and stamps the version file. It stages exactly
  what ships on S3 with a build record (`<name>-<version>-build.json`: the
  commit, run and build number).
- A **deploy** publishes what was staged, by version, without rebuilding it.
  It then tags `v<version>` at the commit the build recorded.

Nothing reaches a registry until you run a deploy, and what it publishes is
exactly what the build checked. A tag records a release; it never triggers
one.

| step | Python | .NET |
| --- | --- | --- |
| build and stage | **Build Library** ([`build-lib.yml`](.github/workflows/build-lib.yml)) | **Build Library (.NET)** ([`build-lib-dotnet.yml`](.github/workflows/build-lib-dotnet.yml)) |
| publish and tag | **Deploy Library to PyPI** ([`deploy-lib.yml`](.github/workflows/deploy-lib.yml)) | **Deploy Library to NuGet (.NET)** ([`deploy-lib-dotnet.yml`](.github/workflows/deploy-lib-dotnet.yml)) |

The desktop app, Dynaface Classic, takes the same two steps after the library,
since it installs the library from PyPI at the version it pins: see
[Releasing the desktop app](#releasing-the-desktop-app).

## One-time setup

Each item is done once, by the account owner. Nothing in the workflows can do
it for you.

1. **AWS secrets.** `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and
   `AWS_DEFAULT_REGION` are already repository secrets. The builds stage on,
   and the deploys read from, `s3://data.heatonresearch.com/library/`.
2. **PyPI.** On pypi.org, open the `dynaface` project, go to **Publishing**,
   and add a GitHub trusted publisher:
   - Owner: `jeffheaton`
   - Repository name: `dynaface`
   - Workflow name: `deploy-lib.yml`
   - Environment name: leave blank

   The deploy then publishes with GitHub's OIDC token, and no PyPI credential
   is stored in the repo. The old `PYPI_API_TOKEN` secret is no longer read.
   Delete it, and revoke the token on PyPI, once the first trusted deploy has
   succeeded:

   ```bash
   gh secret delete PYPI_API_TOKEN -R jeffheaton/dynaface
   ```
3. **NuGet.org.** This is already in place: the 2.0.x deploys used it. Under
   Account → **Trusted Publishing** there is a policy for repository
   `jeffheaton/dynaface` and workflow file `deploy-lib-dotnet.yml`, with no
   environment. The deploy logs in as `jeffheaton` (`NuGet/login`'s `user`),
   who owns both package IDs. If a policy is ever re-created, it starts out
   provisional and expires unless it is used within about a week.
4. **Codecov.** This is already in place. The test workflows upload with
   OIDC under the flags `python` and `dotnet`. [`.github/codecov.yml`](.github/codecov.yml)
   carries each flag forward and makes coverage informational.

## Making a release

1. Set the new version in all four places:
   - `dynaface-lib-python/setup.cfg` (`[metadata] version`)
   - `dynaface-lib-python/dynaface/version.py` (`VERSION`, for local and
     editable installs; the build rewrites this file)
   - `dynaface-lib-dotnet/facial_dll/FacialDll.csproj` (`<Version>`; the
     build packs both packages at this version)
   - `dynaface-lib-dotnet/FacialDll.Onnx/FacialDll.Onnx.csproj` (`<Version>`,
     in lockstep, for local builds)
2. Commit and push to `main`, and let **Test Library** and
   **Test Library (.NET)** go green.
3. Run **Build Library** and **Build Library (.NET)** from the Actions tab,
   on the same commit.
   - **Python:**
     - flake8, mypy, bandit, black and isort run as reports, not gates.
     - The suite runs under pytest, with JUnit results published as a check,
       branch coverage reports, and a 60% threshold that is reported, not
       enforced.
     - It stamps `dynaface/version.py` with the date and run number.
     - It builds the wheel and sdist (the sdist leaves `tests/` out, since the
       tests need `tests_data/`) and checks them with `twine check --strict`.
     - It installs the wheel into an empty venv and imports it.
   - **.NET:**
     - The suite runs once without the models, to prove the model-gated tests
       report Skipped.
     - Then come the `dotnet format` check, which is a gate, and a
       vulnerable-packages report.
     - It stamps `Version.cs` and builds Release.
     - It runs the full suite with the models required.
     - It packs the DLL zip, both `.nupkg` files and their `.snupkg` symbols
       (Source Link pointing at the commit).
   - **Both:** the results are uploaded as run artifacts and staged on S3,
     with the build record.
4. Look at what the builds staged: the artifacts on each run's page, and the
   reports.
5. Run **Deploy Library to PyPI** and **Deploy Library to NuGet (.NET)**,
   each with the version (e.g. `2.1.0`). Each deploy:
   1. downloads its files and build record for that version from S3;
   2. checks that `v<version>` is either untagged or already at the recorded
      commit, and refuses otherwise;
   3. publishes, then tags `v<version>` at that commit.

   Because both builds came from one commit, the second deploy finds the tag
   already where it belongs. Builds from two different commits are refused by
   whichever deploy runs second, before it publishes anything.

A version, once on PyPI or NuGet.org, can't be uploaded again. NuGet's push
skips a duplicate rather than failing; PyPI refuses one. To fix a bad release,
publish the next version.

A build can be rerun at the same version as often as needed: each run
replaces what is staged on S3, so a deploy always publishes the latest build
of that version. Never bump a version just to rebuild. Builds staged before
2026-11-08 (2.0.x, and the first 2.1.0) have no sdist and no build record, so
a deploy refuses them; rebuild before deploying.

## Releasing the desktop app

The desktop app, Dynaface Classic (`dynaface-app/`), shares the library's
version and tag, and its release page: one GitHub Release, `v<version>`, holds
every download of a version, the Unity app's too. Release the library first.

1. Set the version in `dynaface-app/deploy/macos/build.sh` (`DYNAFACE_VERSION`)
   and `dynaface-app/deploy/windows/build.ps1` (both `$script:DYNAFACE_VERSION`
   and `$currentVersion`), and in `dynaface-app/version.py` for local runs. The
   builds install the library from PyPI at this version too.
2. Commit and push to `main`.
3. Run **Build macOS App** ([`build-app-macos.yml`](.github/workflows/build-app-macos.yml))
   and **Build Windows App** ([`build-app-windows.yml`](.github/workflows/build-app-windows.yml))
   from the Actions tab, on the same commit. Each:
   - checks first that its signing and staging secrets are there;
   - stamps `version.py` with the date and run number, and installs the app's
     requirements and the library from PyPI;
   - runs the app's tests in that environment (the `tests` input, on by
     default, can turn them off to get past a problem in a test, never one in
     the app);
   - builds with PyInstaller and signs: Developer ID, notarized and stapled on
     macOS (Apple silicon only); SSL.com eSigner on Windows;
   - stages the zip and its build record on S3.
4. Run **Deploy Desktop App** ([`deploy-app.yml`](.github/workflows/deploy-app.yml))
   with the version (e.g. `2.1.0`). It:
   1. downloads both zips and their build records for that version from S3;
   2. checks that both builds came from one commit, and that `v<version>` is
      on that commit or an earlier one, where the library's deploy usually put
      it; with no tag yet, it tags that commit;
   3. adds both zips to the `v<version>` release, creating it as a draft when
      there is none. It refuses when the release already holds either zip:
      delete that zip from the release first to replace it.
5. Finish the release on GitHub: write the notes (the deploy's run summary
   lists the zips' SHA-256 sums), add the Unity app's downloads, then publish
   it and mark it latest, which moves the README's "Download Latest Version"
   link to it:

   ```bash
   gh release edit v2.1.0 -R jeffheaton/dynaface --draft=false --latest
   ```

The builds can be rerun at the same version as often as needed, and the deploy
takes the latest of each. Builds staged before 2026-11-22 (2.0.x) have no
build record, so the deploy refuses them.

## What is staged

Everything goes under `s3://data.heatonresearch.com/library/`, as public
objects:

| file | from |
| --- | --- |
| `dynaface-<version>-py3-none-any.whl` | Build Library |
| `dynaface-<version>.tar.gz` | Build Library |
| `dynaface-<version>-build.json` | Build Library |
| `Dynaface.<version>.nupkg`, `Dynaface.<version>.snupkg` | Build Library (.NET) |
| `Dynaface.Onnx.<version>.nupkg`, `Dynaface.Onnx.<version>.snupkg` | Build Library (.NET) |
| `Dynaface-<version>-build.json` | Build Library (.NET) |
| `dynaface-dotnet-<version>.zip` (the DLL and PDB, for prebuilt-DLL users) | Build Library (.NET) |

The desktop app's builds go under `s3://data.heatonresearch.com/app/`, beside
the Unity app's, also as public objects:

| file | from |
| --- | --- |
| `dynaface-app-mac-<version>.zip`, `dynaface-app-mac-<version>-build.json` | Build macOS App |
| `dynaface-app-win-<version>.zip`, `dynaface-app-win-<version>-build.json` | Build Windows App |
