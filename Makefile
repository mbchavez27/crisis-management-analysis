.PHONY: setup pipeline preprocess preprocess-youtube preprocess-reddit preprocess-transcripts \
        analysis sentiment-youtube topics-youtube topic-validation sentiment-transcripts topics-transcripts \
        insights sentiment-change summary clean help

# Default target
help:
	@echo "=== Crisis Management Analysis Pipeline ==="
	@echo ""
	@echo "Usage: make <target>"
	@echo ""
	@echo "Setup:"
	@echo "  setup                 Fresh-PC setup (uv sync + calamancy model + NLTK data)"
	@echo ""
	@echo "Pipeline targets:"
	@echo "  pipeline              Run full pipeline (preprocess → analysis → insights)"
	@echo "  preprocess            Run all preprocessing stages"
	@echo "  analysis              Run all analysis stages (depends on preprocess)"
	@echo "  insights              Run all insight stages (depends on analysis)"
	@echo ""
	@echo "Preprocessing targets:"
	@echo "  preprocess-youtube    Preprocess YouTube comments"
	@echo "  preprocess-reddit     Preprocess Reddit posts"
	@echo "  preprocess-transcripts Preprocess video transcripts"
	@echo ""
	@echo "Analysis targets:"
	@echo "  sentiment-youtube     YouTube sentiment analysis"
	@echo "  topics-youtube        YouTube topic modeling"
	@echo "  topic-validation      Cross-phase topic label validation"
	@echo "  sentiment-transcripts Transcript sentiment analysis"
	@echo "  topics-transcripts    Transcript topic modeling"
	@echo ""
	@echo "Insight targets:"
	@echo "  sentiment-change      Sentiment change across phases"
	@echo "  summary               Final summary and visualizations"
	@echo ""
	@echo "Utility targets:"
	@echo "  clean                 Remove all output files"

# === Setup ===

setup:
	@echo "--- Fresh-PC setup ---"
	@bash setup.sh

# Full pipeline
pipeline: preprocess analysis insights
	@echo "=== Pipeline complete ==="

# === Preprocessing ===

preprocess: preprocess-youtube preprocess-reddit preprocess-transcripts
	@echo "=== All preprocessing complete ==="

preprocess-youtube:
	@echo "--- Preprocessing YouTube comments ---"
	@bash 01-preprocessing/run_youtube.sh

preprocess-reddit:
	@echo "--- Preprocessing Reddit posts ---"
	@bash 01-preprocessing/run_reddit.sh

preprocess-transcripts:
	@echo "--- Preprocessing video transcripts ---"
	@bash 01-preprocessing/run_transcripts.sh

# === Analysis ===

analysis: sentiment-youtube topics-youtube topic-validation sentiment-transcripts topics-transcripts
	@echo "=== All analysis complete ==="

sentiment-youtube:
	@echo "--- YouTube sentiment analysis ---"
	@bash 02-analysis/youtube/run_sentiment.sh

topics-youtube:
	@echo "--- YouTube topic modeling ---"
	@bash 02-analysis/youtube/run_topics.sh

topic-validation:
	@echo "--- Topic validation (cross-phase review) ---"
	@bash 02-analysis/youtube/run_topic_validation.sh

sentiment-transcripts:
	@echo "--- Transcript sentiment analysis ---"
	@bash 02-analysis/transcripts/run_sentiment.sh

topics-transcripts:
	@echo "--- Transcript topic modeling ---"
	@bash 02-analysis/transcripts/run_topics.sh

# === Insights ===

insights: sentiment-change summary
	@echo "=== All insights complete ==="

sentiment-change:
	@echo "--- Sentiment change across phases ---"
	@bash 02-analysis/youtube/run_sentiment_change.sh

stats:
	@echo "--- Statistical analysis (merge + models) ---"
	@bash 04-statistical-analysis/run_stats.sh

summary:
	@echo "--- Generating summary ---"
	@bash 05-summary/run_summary.sh

# === Cleanup ===

clean:
	@echo "--- Cleaning outputs ---"
	rm -f 01-preprocessing/output/ateneo-drowning/youtube/*.csv
	rm -f 01-preprocessing/output/ateneo-drowning/reddit/*.csv
	rm -f 01-preprocessing/output/ateneo-drowning/transcripts/*.csv
	rm -f 02-analysis/output/ateneo-drowning/youtube-comments/sentiment-analysis/*.csv
	rm -f 02-analysis/output/ateneo-drowning/youtube-comments/topic-modeling/*.csv
	rm -f 02-analysis/output/ateneo-drowning/transcripts/sentiment-analysis/*.csv
	rm -f 02-analysis/output/ateneo-drowning/transcripts/topic-modeling/*.csv
	rm -f 02-analysis/output/ateneo-drowning/reddit/sentiment-analysis/*.csv
	rm -f 02-analysis/output/ateneo-drowning/reddit/topic-modeling/*.csv
	rm -f outputs/figures/01-preprocessing/*.png
	rm -f outputs/figures/02-analysis/youtube/*.png
	rm -f outputs/figures/02-analysis/transcripts/*.png
	rm -f outputs/figures/05-summary/*.png
	@echo "=== Outputs cleaned ==="
