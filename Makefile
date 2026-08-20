.PHONY: build serve bench quality verify model-fetch model-verify manifest clean

build:           ## build venv + apply patches + kernels wheel
	bash scripts/build.sh

serve:           ## serve TP2 (override: TP=1, MTP=4, INT8=off, KV=fp8, CTX=...)
	bash scripts/serve.sh $(ARGS)

bench:           ## cold strict suite + reference comparison
	bash scripts/bench-strict.sh

quality:         ## quality gate (canary must answer 14)
	bash scripts/quality-gate.sh

verify:          ## ops + canary + cache-zero + env manifest
	bash scripts/verify-install.sh

model-fetch:
	bash scripts/model.sh fetch
model-verify:
	bash scripts/model.sh verify

manifest:        ## print the installed environment manifest
	[[ -f env-manifest.json ]] && cat env-manifest.json || echo "run verify first"

clean:
	rm -rf .venv dist src env-manifest.json
