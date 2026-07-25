-- Runs automatically ONE TIME, the first time the Postgres container starts
-- with an empty data volume (Postgres's official image does this for any
-- .sql file mounted into /docker-entrypoint-initdb.d/).

-- uuid-ossp gives us functions like uuid_generate_v4() for generating
-- unique IDs — used later when we create database tables (Module 3).
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";