# ==========================================
# BoolForge tutorial pipeline
# ==========================================

# Source tutorials
PY_TUTORIALS := $(wildcard src/tutorial*.py)

# Output directories
IPYNB_DIR := tutorials_ipynb
HTML_DIR := tutorials_html

# Generated notebooks
IPYNB_FILES := $(patsubst src/%.py,$(IPYNB_DIR)/%.ipynb,$(PY_TUTORIALS))

# Generated HTML files
HTML_FILES := $(patsubst $(IPYNB_DIR)/%.ipynb,$(HTML_DIR)/%.html,$(IPYNB_FILES))

# Execution timeout (seconds)
TIMEOUT := 300

.PHONY: tutorials html pdf supplement clean print

# ------------------------------------------
# Main target
# ------------------------------------------

tutorials: $(IPYNB_FILES)
	@echo "All tutorials converted and executed successfully."

# ------------------------------------------
# Directory creation
# ------------------------------------------

$(IPYNB_DIR):
	mkdir -p $(IPYNB_DIR)

$(HTML_DIR):
	mkdir -p $(HTML_DIR)

# ------------------------------------------
# Notebook generation
# ------------------------------------------

$(IPYNB_DIR)/%.ipynb: src/%.py | $(IPYNB_DIR)
	@echo "Converting $< → $@"
	jupytext --to notebook $< --output $@
	@echo "Executing $@"
	jupyter nbconvert \
		--execute \
		--to notebook \
		--inplace \
		--ExecutePreprocessor.timeout=$(TIMEOUT) \
		$@

# ------------------------------------------
# HTML generation
# ------------------------------------------

$(HTML_DIR)/%.html: $(IPYNB_DIR)/%.ipynb | $(HTML_DIR)
	@echo "Rendering $@"
	jupyter nbconvert \
		--to html \
		$< \
		--output-dir=$(HTML_DIR)

html: $(HTML_FILES)
	@echo "Created HTML files."

# ------------------------------------------
# PDF generation
# ------------------------------------------

pdf: tutorials
	./build_tutorials.sh standalone

supplement: tutorials
	./build_tutorials.sh supplement

# ------------------------------------------
# Diagnostics
# ------------------------------------------

print:
	@echo "PY_TUTORIALS:"
	@echo $(PY_TUTORIALS)
	@echo
	@echo "IPYNB_FILES:"
	@echo $(IPYNB_FILES)
	@echo
	@echo "HTML_FILES:"
	@echo $(HTML_FILES)

# ------------------------------------------
# Cleaning
# ------------------------------------------

clean:
	rm -rf $(IPYNB_DIR)
	rm -rf $(HTML_DIR)
