#!/bin/sh
# Creates the read-only role Grafana connects as. Runs after init.sql on an empty
# data volume. The password comes from GRAFANA_DB_PASSWORD (see .env.example).
set -eu

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
     -v reader_password="$GRAFANA_DB_PASSWORD" <<'SQL'
CREATE ROLE grafana_reader LOGIN PASSWORD :'reader_password';
SELECT format('GRANT CONNECT ON DATABASE %I TO grafana_reader', current_database()) \gexec
GRANT USAGE ON SCHEMA public TO grafana_reader;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO grafana_reader;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO grafana_reader;
SQL
