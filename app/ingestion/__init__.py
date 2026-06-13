"""
Ingestion package.  Run loaders via the CLI:

    # Full UK + EU (recommended first run)
    python -m app.ingestion

    # UK only
    python -m app.ingestion --jurisdiction uk

    # EU with a local CSV file (if auto-discovery fails)
    python -m app.ingestion --jurisdiction eu --eu-csv-path /path/to/taric.csv
"""
