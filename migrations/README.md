# Database Migrations

This directory contains database migration scripts for the Discord SUI Analyzer project.

## Migration Overview

**IMPORTANT**: Run migrations in the correct order:
1. `add_duplicate_tracking.py` - Adds duplicate solution tracking
2. `add_layered_processing.py` - Implements complete layered processing strategy

## Layered Processing Migration

The `add_layered_processing.py` migration implements the complete layered processing strategy where original messages remain unchanged and each processing step adds metadata layers.

### What it does:

1. **New Tables:**
   - `message_processing`: Tracks each processing step applied to individual messages
   - `message_annotations`: Stores message-level classifications and annotations
   - `processing_pipeline`: Defines and tracks processing pipeline configuration

2. **Enhanced Existing Tables:**
   - **Messages**: Adds `processing_status`, `last_processed_at`, `processing_version`
   - **Threads**: Adds `processing_history`, `confidence_scores`, `processing_metadata`  
   - **Solutions**: Adds `extraction_metadata`, `processing_steps`, `source_messages`

3. **Processing Transparency Features:**
   - Complete audit trail of all processing steps
   - Confidence scores for each processing decision
   - Version tracking for algorithm changes
   - Message-level annotations for classification

4. **Default Processing Pipeline:**
   - Initializes standard 6-step processing pipeline
   - Configurable pipeline steps with metadata

### Usage:

#### Check if database is ready:
```bash
cd migrations
python add_layered_processing.py --check
```

#### Run the migration:
```bash
cd migrations
python add_layered_processing.py
```

#### Verify migration:
```bash
cd migrations
python add_layered_processing.py --verify
```

#### Rollback the migration:
```bash
cd migrations
python add_layered_processing.py --rollback
```

## Duplicate Tracking Migration

The `add_duplicate_tracking.py` migration adds duplicate solution tracking functionality to the database.

### What it does:

1. **Adds new columns to `solutions` table:**
   - `is_duplicate`: Boolean flag indicating if a solution is a duplicate
   - `duplicate_count`: Count of duplicates pointing to this solution

2. **Creates `solution_duplicates` table:**
   - Tracks relationships between duplicate solutions and their originals
   - Stores similarity scores and admin review status
   - Includes admin review fields (reviewed_by, reviewed_at, notes)

3. **Creates performance indexes:**
   - Indexes for efficient duplicate queries
   - Unique constraint to prevent duplicate duplicate records

### Usage:

#### Check if database is ready:
```bash
cd migrations
python add_duplicate_tracking.py --check
```

#### Run the migration:
```bash
cd migrations
python add_duplicate_tracking.py
```

#### Rollback the migration:
```bash
cd migrations
python add_duplicate_tracking.py --rollback
```

### Requirements:

- Database must exist with existing `solutions` and `threads` tables
- PostgreSQL database (uses SERIAL for auto-increment)
- Proper environment variables set (PEERA_DB_URL or config.yaml)

### Migration Status:

The script automatically tracks migration status using a `migrations` table. It will skip re-applying the migration if it has already been run.

### Error Handling:

- All operations are wrapped in transactions
- Automatic rollback on failure
- Comprehensive logging for debugging

### Post-Migration:

After running this migration, the following new admin endpoints will be available:

- `GET /admin/duplicates` - List pending duplicate reviews
- `GET /admin/duplicates/{solution_id}` - Get duplicate chain for a solution
- `PUT /admin/duplicates/{duplicate_id}/review` - Review and update duplicate status
- `GET /admin/frequency-analysis` - Get problem frequency analysis
- `GET /admin/solutions/{topic_id}/duplicates` - Get duplicates by topic ID
- `GET /admin/duplicate-statistics` - Get duplicate statistics
- `POST /admin/recalculate-duplicate-counts` - Recalculate duplicate counts

### Verification:

After migration, verify the new tables and columns exist:

```sql
-- Check new columns in solutions table
\d solutions

-- Check new solution_duplicates table
\d solution_duplicates

-- Verify indexes
\di solution_duplicates

-- Check migration record
SELECT * FROM migrations WHERE name = 'add_duplicate_tracking';
```