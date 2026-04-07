# Local SonarQube Scan

Use this runbook to reproduce the local SonarQube scan for this repository.

## Scope

- Run unit tests with coverage in the `conda` env `exllama`
- Generate `coverage.xml`
- Run a local SonarQube server in Docker
- Run the Dockerized Sonar scanner against the local server

## Files used by the scan

- `sonar-project.properties`
- `coverage.xml`

`sonar-project.properties` already points SonarQube at:

- source code under `src/gallama`
- tests under `tests`
- coverage report `coverage.xml`

## 1. Start local SonarQube

If the container does not exist yet:

```bash
docker pull sonarqube:lts-community
docker run -d --name sonarqube-local -p 9000:9000 sonarqube:lts-community
```

If it already exists but is stopped:

```bash
docker start sonarqube-local
```

Wait until the server is up:

```bash
curl -sS http://localhost:9000/api/system/status
```

Proceed only when the returned status is `UP`.

## 2. Create or reuse a Sonar token

Generate a token from the local server:

```bash
curl -sS -u admin:admin -X POST \
  'http://localhost:9000/api/user_tokens/generate' \
  -d 'name=local-scan-token'
```

Use the returned token as `SONAR_TOKEN`.

If the default admin password no longer works, use an existing local token instead.

## 3. Run unit tests with coverage

Important: do not run `pytest tests/unit` in one process for coverage in this repo.
Some test files monkeypatch `sys.modules["fastapi"]`, which can bleed across files during a single run.

Run the unit files one by one and append coverage:

```bash
PYTHONPATH=/home/remichu/work/ML/gallama/src:/home/remichu/work/ML/gallama/tests \
conda run -n exllama python -m coverage erase

for test_file in $(find tests/unit -maxdepth 1 -type f -name 'test_*.py' | sort); do
  echo "==> ${test_file}"
  PYTHONPATH=/home/remichu/work/ML/gallama/src:/home/remichu/work/ML/gallama/tests \
  conda run -n exllama python -m coverage run --append -m pytest -q "$test_file"
done

PYTHONPATH=/home/remichu/work/ML/gallama/src:/home/remichu/work/ML/gallama/tests \
conda run -n exllama python -m coverage xml -o coverage.xml

PYTHONPATH=/home/remichu/work/ML/gallama/src:/home/remichu/work/ML/gallama/tests \
conda run -n exllama python -m coverage report -m
```

Expected result:

- `coverage.xml` is created in the repo root
- coverage summary prints in the terminal

## 4. Run the Dockerized scanner

Replace `<SONAR_TOKEN>` with the token from step 2:

```bash
docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -e SONAR_HOST_URL=http://host.docker.internal:9000 \
  -e SONAR_TOKEN=<SONAR_TOKEN> \
  -v /home/remichu/work/ML/gallama:/usr/src \
  sonarsource/sonar-scanner-cli:latest
```

Expected result:

- scanner prints `ANALYSIS SUCCESSFUL`
- dashboard URL looks like `http://localhost:9000/dashboard?id=gallama`

## 5. Verify scan results

Project measures:

```bash
curl -sS -u '<SONAR_TOKEN>:' \
  'http://localhost:9000/api/measures/component?component=gallama&metricKeys=coverage,bugs,reliability_rating,security_rating,security_review_rating,alert_status,vulnerabilities,security_hotspots'
```

Open bugs:

```bash
curl -sS -u '<SONAR_TOKEN>:' \
  'http://localhost:9000/api/issues/search?componentKeys=gallama&types=BUG&resolved=false&ps=100'
```

## Notes

- SonarQube `9.9.x` warns that Python `3.11` is treated as `3.10`. The scan still runs.
- The scanner may warn about missing SCM blame for locally modified or untracked files. That is expected in a dirty worktree.
- The scanner may warn about the Cobertura `<source>` path in `coverage.xml`. Coverage can still import successfully.
