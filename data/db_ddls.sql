-- #----------------------------#
-- # Author: Surjit Das
-- # Email: surjitdas@gmail.com
-- # Program: artmind
-- #----------------------------#

PRAGMA foreign_keys = off;
BEGIN TRANSACTION;

-- Table: external_kbs
CREATE TABLE external_kbs (
    item                TEXT,
    list_wdInstance     TEXT,
    list_wikiDataClass  TEXT,
    list_dbPediaType    TEXT,
    list_conceptNetType TEXT,
    ts                  TIMESTAMP
);


-- Table: sentences
CREATE TABLE sentences (
    sentence_uuid   TEXT,
    TYPE            TEXT,
    NER_type        TEXT,
    item            TEXT,
    token_dep       TEXT,
    token_pos       TEXT,
    token_head_text TEXT,
    token_lemma     TEXT,
    ts              TIMESTAMP
);


-- View: vw_sentences
CREATE VIEW vw_sentences AS
    SELECT a.sentence_uuid,
           a.TYPE TYPE,
           a.NER_type NER_type,
           a.item item,
           a.token_dep token_dep,
           a.token_pos token_pos,
           a.token_head_text token_head_text,
           a.token_lemma token_lemma,
           a.ts ts_sentences,
           b.list_wdInstance list_wdInstance,
           b.list_wikiDataClass list_wikiDataClass,
           b.list_dbPediaType list_dbPediaType,
           b.list_conceptNetType list_conceptNetType,
           b.ts ts_kbs
      FROM sentences a
           LEFT JOIN
           external_kbs b ON a.item = b.item;


COMMIT TRANSACTION;
PRAGMA foreign_keys = on;
