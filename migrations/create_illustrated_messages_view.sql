-- Create illustrated_messages view that replicates the illustrated_message function logic
-- This view formats messages with admin/user prefixes and handles replies

-- Drop the view if it exists
DROP VIEW IF EXISTS illustrated_messages;

-- Create the illustrated_messages view
CREATE OR REPLACE VIEW public.illustrated_messages
 AS
 SELECT m.message_id,
    m.content,
    m.datetime,
    m.referenced_message_id,
    m.parent_id,
    m.thread_id,
    a.author_id,
    a.author_name,
    a.author_type,
    concat(to_char(m.datetime, 'YYYY-MM-DD HH24:MI:SS '::text), ' ',
        CASE
            WHEN (EXISTS ( SELECT 1
               FROM admins ad
              WHERE ad.author_id::text = m.author_id::text)) THEN concat('Admin ', a.author_name)
            ELSE a.author_name
        END, ': ',
        CASE
            WHEN (m.referenced_message_id IS NOT NULL) and (m.referenced_message_id<>'nan') THEN concat('reply to ',
            CASE
                WHEN (EXISTS ( SELECT 1
                   FROM admins ad
                  WHERE ad.author_id::text = ref_author.author_id::text)) THEN concat('Admin ', ref_author.author_name)
                ELSE ref_author.author_name
            END, ' - ')
            ELSE ''::text
        END, COALESCE(regexp_replace(m.content, '<@(\d+)>'::text, ('<# '::text ||
        CASE
            WHEN (EXISTS ( SELECT 1
               FROM admins ad
              WHERE ad.author_id::text = "substring"(m.content, '<@(\d+)>'::text))) THEN concat('Admin ', COALESCE(tagged_author.author_name, "substring"(m.content, '<@(\d+)>'::text)::character varying))
            ELSE concat(COALESCE(tagged_author.author_name, "substring"(m.content, '<@(\d+)>'::text)::character varying))
        END) || '>'::text, 'g'::text), '<empty>'::text)) AS illustrated_message
   FROM messages m
     LEFT JOIN authors a ON m.author_id::text = a.author_id::text
     LEFT JOIN messages ref_msg ON m.referenced_message_id::text = ref_msg.message_id::text
     LEFT JOIN authors ref_author ON ref_msg.author_id::text = ref_author.author_id::text
     LEFT JOIN authors tagged_author ON tagged_author.author_id::text = "substring"(m.content, '<@(\d+)>'::text);

ALTER TABLE public.illustrated_messages
    OWNER TO postgres;
