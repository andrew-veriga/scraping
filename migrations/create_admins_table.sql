-- Create admins table and insert admin IDs
-- This script creates a table to store Discord admin user IDs

-- Create the admins table
CREATE TABLE IF NOT EXISTS admins (
    author_id VARCHAR(50) PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Insert admin IDs from the Discord server
INSERT INTO admins (author_id) VALUES 
    ('862550907349893151'),
    ('466815633347313664'),
    ('997105563123064892'),
    ('457962750644060170')
ON CONFLICT (author_id) DO NOTHING;

-- Verify the data was inserted
SELECT * FROM admins ORDER BY created_at;
