SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;
CREATE EXTENSION IF NOT EXISTS "pgsodium";
COMMENT ON SCHEMA "public" IS 'standard public schema';
CREATE EXTENSION IF NOT EXISTS "pg_graphql" WITH SCHEMA "graphql";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements" WITH SCHEMA "extensions";
CREATE EXTENSION IF NOT EXISTS "pg_trgm" WITH SCHEMA "public";
CREATE EXTENSION IF NOT EXISTS "pgcrypto" WITH SCHEMA "extensions";
CREATE EXTENSION IF NOT EXISTS "pgjwt" WITH SCHEMA "extensions";
CREATE EXTENSION IF NOT EXISTS "supabase_vault" WITH SCHEMA "vault";
CREATE EXTENSION IF NOT EXISTS "uuid-ossp" WITH SCHEMA "extensions";
CREATE EXTENSION IF NOT EXISTS "vector" WITH SCHEMA "public";
CREATE TYPE "public"."attention_level" AS ENUM (
    'high',
    'medium',
    'low'
);
ALTER TYPE "public"."attention_level" OWNER TO "postgres";
CREATE TYPE "public"."conceptual_similarity_level" AS ENUM (
    'identical',
    'high degree',
    'medium degree',
    'low degree',
    'dissimilar',
    'neutral'
);
ALTER TYPE "public"."conceptual_similarity_level" OWNER TO "postgres";
CREATE TYPE "public"."confusion_type" AS ENUM (
    'direct',
    'indirect',
    'both'
);
ALTER TYPE "public"."confusion_type" OWNER TO "postgres";
CREATE TYPE "public"."distinctive_character_level" AS ENUM (
    'high',
    'medium',
    'low',
    'enhanced'
);
ALTER TYPE "public"."distinctive_character_level" OWNER TO "postgres";
CREATE TYPE "public"."jurisdiction_type" AS ENUM (
    'UKIPO',
    'EUIPO'
);
ALTER TYPE "public"."jurisdiction_type" OWNER TO "postgres";
CREATE TYPE "public"."opposition_outcome" AS ENUM (
    'successful',
    'partially successful',
    'unsuccessful'
);
ALTER TYPE "public"."opposition_outcome" OWNER TO "postgres";
CREATE TYPE "public"."proof_of_use_status" AS ENUM (
    'use_proven',
    'use_not_proven',
    'not_applicable'
);
ALTER TYPE "public"."proof_of_use_status" OWNER TO "postgres";
CREATE TYPE "public"."similarity_level" AS ENUM (
    'identical',
    'high degree',
    'medium degree',
    'low degree',
    'dissimilar'
);
ALTER TYPE "public"."similarity_level" OWNER TO "postgres";
CREATE OR REPLACE FUNCTION "public"."begin_transaction"() RETURNS "void"
    LANGUAGE "plpgsql"
    AS $$
BEGIN
  -- Start a transaction
  -- This is just a placeholder since Supabase/PostgreSQL starts transactions with BEGIN statement
  -- which is handled by the connection itself
END;
$$;
ALTER FUNCTION "public"."begin_transaction"() OWNER TO "postgres";
CREATE OR REPLACE FUNCTION "public"."commit_transaction"() RETURNS "void"
    LANGUAGE "plpgsql"
    AS $$
BEGIN
  -- Commit a transaction
  -- This is just a placeholder since Supabase/PostgreSQL commits with COMMIT statement
  -- which is handled by the connection itself
END;
$$;
ALTER FUNCTION "public"."commit_transaction"() OWNER TO "postgres";
CREATE OR REPLACE FUNCTION "public"."find_similar_cases"("mark_text" "text", "limit_count" integer DEFAULT 5) RETURNS TABLE("case_reference" "text", "similarity_score" double precision)
    LANGUAGE "plpgsql"
    AS $$
BEGIN
  RETURN QUERY
  SELECT 
    tc.case_reference,
    MAX(similarity(lower(mark_text), lower(am.mark))) as score
  FROM 
    trademark_cases tc
    JOIN applicant_marks am ON tc.case_reference = am.case_reference
  GROUP BY 
    tc.case_reference
  ORDER BY 
    score DESC
  LIMIT limit_count;
END;
$$;
ALTER FUNCTION "public"."find_similar_cases"("mark_text" "text", "limit_count" integer) OWNER TO "postgres";
CREATE OR REPLACE FUNCTION "public"."find_similar_goods_services"("search_term" "text", "class_num" integer, "limit_count" integer DEFAULT 10) RETURNS TABLE("id" integer, "term" "text", "nice_class" integer, "similarity_score" double precision)
    LANGUAGE "plpgsql"
    AS $$
BEGIN
  RETURN QUERY
  SELECT 
    gs.id, 
    gs.term, 
    gs.nice_class, 
    similarity(lower(search_term), lower(gs.term)) as sim_score
  FROM 
    goods_services gs
  WHERE 
    gs.nice_class = class_num
  ORDER BY 
    sim_score DESC
  LIMIT limit_count;
END;
$$;
ALTER FUNCTION "public"."find_similar_goods_services"("search_term" "text", "class_num" integer, "limit_count" integer) OWNER TO "postgres";
CREATE OR REPLACE FUNCTION "public"."find_similar_term"("input_term" "text", "input_nice_class" integer, "threshold" double precision) RETURNS TABLE("id" integer)
    LANGUAGE "plpgsql"
    AS $$
BEGIN
  RETURN QUERY
  SELECT gs.id 
  FROM goods_services gs
  WHERE gs.nice_class = input_nice_class
    AND similarity(lower(gs.term), lower(input_term)) > threshold
  ORDER BY similarity(lower(gs.term), lower(input_term)) DESC
  LIMIT 1;
END;
$$;
ALTER FUNCTION "public"."find_similar_term"("input_term" "text", "input_nice_class" integer, "threshold" double precision) OWNER TO "postgres";
CREATE OR REPLACE FUNCTION "public"."get_goods_services_comparisons"("limit_count" integer DEFAULT 10000) RETURNS TABLE("similarity" "text", "term1" "text", "class1" integer, "term2" "text", "class2" integer, "id1" integer, "id2" integer)
    LANGUAGE "plpgsql"
    AS $$
BEGIN
  RETURN QUERY
  SELECT 
    gsc.similarity,
    gs1.term as term1,
    gs1.nice_class as class1,
    gs2.term as term2,
    gs2.nice_class as class2,
    gs1.id as id1,
    gs2.id as id2
  FROM 
    goods_services_comparisons gsc
  JOIN 
    goods_services gs1 ON gsc.goods_services_1_id = gs1.id
  JOIN 
    goods_services gs2 ON gsc.goods_services_2_id = gs2.id
  LIMIT limit_count;
END;
$$;
ALTER FUNCTION "public"."get_goods_services_comparisons"("limit_count" integer) OWNER TO "postgres";
CREATE OR REPLACE FUNCTION "public"."predict_similarity"("term1" "text", "nice_class1" integer, "term2" "text", "nice_class2" integer) RETURNS "text"
    LANGUAGE "plpgsql"
    AS $$
DECLARE
    text_similarity double precision;
    similarity_category text;
    similar_comparison RECORD;
BEGIN
    -- First look for exact matches
    SELECT gsc.similarity::text INTO similarity_category
    FROM goods_services gs1
    JOIN goods_services_comparisons gsc ON gs1.id = gsc.goods_services_1_id
    JOIN goods_services gs2 ON gs2.id = gsc.goods_services_2_id
    WHERE 
        gs1.nice_class = nice_class1 AND 
        gs2.nice_class = nice_class2 AND
        (
            (LOWER(gs1.term) = LOWER(term1) AND LOWER(gs2.term) = LOWER(term2)) OR
            (LOWER(gs1.term) = LOWER(term2) AND LOWER(gs2.term) = LOWER(term1))
        )
    LIMIT 1;

    -- If exact match found, return it
    IF similarity_category IS NOT NULL THEN
        RETURN similarity_category;
    END IF;

    -- Look for semantically similar comparisons
    -- We'll use a combination of text similarity to find relevant comparisons
    SELECT gsc.similarity::text INTO similarity_category
    FROM goods_services gs1
    JOIN goods_services_comparisons gsc ON gs1.id = gsc.goods_services_1_id
    JOIN goods_services gs2 ON gs2.id = gsc.goods_services_2_id
    WHERE 
        gs1.nice_class = nice_class1 AND 
        gs2.nice_class = nice_class2 AND
        (
            -- Look for similar terms using pg_trgm similarity function
            (
                similarity(LOWER(gs1.term), LOWER(term1)) > 0.4 AND 
                similarity(LOWER(gs2.term), LOWER(term2)) > 0.4
            )
            OR
            (
                similarity(LOWER(gs1.term), LOWER(term2)) > 0.4 AND 
                similarity(LOWER(gs2.term), LOWER(term1)) > 0.4
            )
        )
    ORDER BY 
        GREATEST(
            similarity(LOWER(gs1.term), LOWER(term1)) + similarity(LOWER(gs2.term), LOWER(term2)),
            similarity(LOWER(gs1.term), LOWER(term2)) + similarity(LOWER(gs2.term), LOWER(term1))
        ) DESC
    LIMIT 1;

    -- If we found a similar comparison, return it
    IF similarity_category IS NOT NULL THEN
        RETURN similarity_category;
    END IF;
    
    -- If no similar comparisons found, fall back to direct text similarity
    SELECT similarity(LOWER(term1), LOWER(term2)) INTO text_similarity;
    
    -- Map the numerical score to a category
    IF text_similarity >= 0.9 THEN
        similarity_category := 'identical';
    ELSIF text_similarity >= 0.7 THEN
        similarity_category := 'high degree';
    ELSIF text_similarity >= 0.5 THEN
        similarity_category := 'medium degree';
    ELSIF text_similarity >= 0.3 THEN
        similarity_category := 'low degree';
    ELSE
        similarity_category := 'dissimilar';
    END IF;
    
    RETURN similarity_category;
END;
$$;
ALTER FUNCTION "public"."predict_similarity"("term1" "text", "nice_class1" integer, "term2" "text", "nice_class2" integer) OWNER TO "postgres";
CREATE OR REPLACE FUNCTION "public"."rollback_transaction"() RETURNS "void"
    LANGUAGE "plpgsql"
    AS $$
BEGIN
  -- Rollback a transaction
  -- This is just a placeholder since Supabase/PostgreSQL rolls back with ROLLBACK statement
  -- which is handled by the connection itself
END;
$$;
ALTER FUNCTION "public"."rollback_transaction"() OWNER TO "postgres";
CREATE OR REPLACE FUNCTION "public"."update_timestamp"() RETURNS "trigger"
    LANGUAGE "plpgsql"
    AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$;
ALTER FUNCTION "public"."update_timestamp"() OWNER TO "postgres";
SET default_tablespace = '';
SET default_table_access_method = "heap";
CREATE TABLE IF NOT EXISTS "public"."applicant_goods_services" (
    "id" integer NOT NULL,
    "mark_id" integer,
    "goods_services_id" integer,
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE "public"."applicant_goods_services" OWNER TO "postgres";
CREATE SEQUENCE IF NOT EXISTS "public"."applicant_goods_services_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER TABLE "public"."applicant_goods_services_id_seq" OWNER TO "postgres";
ALTER SEQUENCE "public"."applicant_goods_services_id_seq" OWNED BY "public"."applicant_goods_services"."id";
CREATE TABLE IF NOT EXISTS "public"."applicant_marks" (
    "id" integer NOT NULL,
    "case_reference" "text",
    "mark" "text" NOT NULL,
    "mark_is_figurative" boolean DEFAULT false NOT NULL,
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE "public"."applicant_marks" OWNER TO "postgres";
CREATE SEQUENCE IF NOT EXISTS "public"."applicant_marks_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER TABLE "public"."applicant_marks_id_seq" OWNER TO "postgres";
ALTER SEQUENCE "public"."applicant_marks_id_seq" OWNED BY "public"."applicant_marks"."id";
CREATE TABLE IF NOT EXISTS "public"."decision_rationales" (
    "id" integer NOT NULL,
    "case_reference" "text",
    "key_factors" "text"[] NOT NULL,
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE "public"."decision_rationales" OWNER TO "postgres";
CREATE SEQUENCE IF NOT EXISTS "public"."decision_rationales_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER TABLE "public"."decision_rationales_id_seq" OWNER TO "postgres";
ALTER SEQUENCE "public"."decision_rationales_id_seq" OWNED BY "public"."decision_rationales"."id";
CREATE TABLE IF NOT EXISTS "public"."goods_services" (
    "id" integer NOT NULL,
    "term" "text" NOT NULL,
    "nice_class" integer NOT NULL,
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE "public"."goods_services" OWNER TO "postgres";
CREATE TABLE IF NOT EXISTS "public"."goods_services_comparisons" (
    "id" integer NOT NULL,
    "goods_services_1_id" integer,
    "goods_services_2_id" integer,
    "similarity" "public"."similarity_level" NOT NULL,
    "market_context" "jsonb",
    "case_reference" "text",
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE "public"."goods_services_comparisons" OWNER TO "postgres";
CREATE SEQUENCE IF NOT EXISTS "public"."goods_services_comparisons_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER TABLE "public"."goods_services_comparisons_id_seq" OWNER TO "postgres";
ALTER SEQUENCE "public"."goods_services_comparisons_id_seq" OWNED BY "public"."goods_services_comparisons"."id";
CREATE SEQUENCE IF NOT EXISTS "public"."goods_services_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER TABLE "public"."goods_services_id_seq" OWNER TO "postgres";
ALTER SEQUENCE "public"."goods_services_id_seq" OWNED BY "public"."goods_services"."id";
CREATE TABLE IF NOT EXISTS "public"."mark_comparisons" (
    "id" integer NOT NULL,
    "applicant_mark_id" integer,
    "opponent_mark_id" integer,
    "case_reference" "text",
    "visual_similarity" "public"."similarity_level",
    "aural_similarity" "public"."similarity_level",
    "conceptual_similarity" "public"."conceptual_similarity_level",
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE "public"."mark_comparisons" OWNER TO "postgres";
CREATE SEQUENCE IF NOT EXISTS "public"."mark_comparisons_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER TABLE "public"."mark_comparisons_id_seq" OWNER TO "postgres";
ALTER SEQUENCE "public"."mark_comparisons_id_seq" OWNED BY "public"."mark_comparisons"."id";
CREATE TABLE IF NOT EXISTS "public"."opponent_goods_services" (
    "id" integer NOT NULL,
    "mark_id" integer,
    "goods_services_id" integer,
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE "public"."opponent_goods_services" OWNER TO "postgres";
CREATE SEQUENCE IF NOT EXISTS "public"."opponent_goods_services_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER TABLE "public"."opponent_goods_services_id_seq" OWNER TO "postgres";
ALTER SEQUENCE "public"."opponent_goods_services_id_seq" OWNED BY "public"."opponent_goods_services"."id";
CREATE TABLE IF NOT EXISTS "public"."opponent_marks" (
    "id" integer NOT NULL,
    "case_reference" "text",
    "mark" "text" NOT NULL,
    "mark_is_figurative" boolean DEFAULT false NOT NULL,
    "registration_number" "text",
    "filing_date" "date",
    "registration_date" "date",
    "priority_date" "date",
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE "public"."opponent_marks" OWNER TO "postgres";
CREATE SEQUENCE IF NOT EXISTS "public"."opponent_marks_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER TABLE "public"."opponent_marks_id_seq" OWNER TO "postgres";
ALTER SEQUENCE "public"."opponent_marks_id_seq" OWNED BY "public"."opponent_marks"."id";
CREATE TABLE IF NOT EXISTS "public"."opposition_grounds" (
    "id" integer NOT NULL,
    "case_reference" "text",
    "ground" "text" NOT NULL,
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE "public"."opposition_grounds" OWNER TO "postgres";
CREATE SEQUENCE IF NOT EXISTS "public"."opposition_grounds_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER TABLE "public"."opposition_grounds_id_seq" OWNER TO "postgres";
ALTER SEQUENCE "public"."opposition_grounds_id_seq" OWNED BY "public"."opposition_grounds"."id";
CREATE TABLE IF NOT EXISTS "public"."precedents_cited" (
    "id" integer NOT NULL,
    "decision_rationale_id" integer,
    "title" "text" NOT NULL,
    "case_reference" "text",
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE "public"."precedents_cited" OWNER TO "postgres";
CREATE SEQUENCE IF NOT EXISTS "public"."precedents_cited_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER TABLE "public"."precedents_cited_id_seq" OWNER TO "postgres";
ALTER SEQUENCE "public"."precedents_cited_id_seq" OWNED BY "public"."precedents_cited"."id";
CREATE TABLE IF NOT EXISTS "public"."trademark_cases" (
    "case_reference" "text" NOT NULL,
    "application_number" "text",
    "applicant_name" "text",
    "opponent_name" "text",
    "proof_of_use_requested" boolean,
    "proof_of_use_outcome" "public"."proof_of_use_status",
    "distinctive_character" "public"."distinctive_character_level",
    "average_consumer_attention" "public"."attention_level",
    "opposition_outcome" "public"."opposition_outcome",
    "decision_maker" "text",
    "jurisdiction" "public"."jurisdiction_type",
    "case_pdf_path" "text",
    "metadata" "jsonb",
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE "public"."trademark_cases" OWNER TO "postgres";
CREATE TABLE IF NOT EXISTS "public"."vector_embeddings" (
    "id" integer NOT NULL,
    "entity_type" character varying(50) NOT NULL,
    "entity_id" integer NOT NULL,
    "embedding" "public"."vector"(1536) NOT NULL,
    "created_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP,
    "updated_at" timestamp with time zone DEFAULT CURRENT_TIMESTAMP
);
ALTER TABLE "public"."vector_embeddings" OWNER TO "postgres";
CREATE SEQUENCE IF NOT EXISTS "public"."vector_embeddings_id_seq"
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
ALTER TABLE "public"."vector_embeddings_id_seq" OWNER TO "postgres";
ALTER SEQUENCE "public"."vector_embeddings_id_seq" OWNED BY "public"."vector_embeddings"."id";
ALTER TABLE ONLY "public"."applicant_goods_services" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."applicant_goods_services_id_seq"'::"regclass");
ALTER TABLE ONLY "public"."applicant_marks" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."applicant_marks_id_seq"'::"regclass");
ALTER TABLE ONLY "public"."decision_rationales" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."decision_rationales_id_seq"'::"regclass");
ALTER TABLE ONLY "public"."goods_services" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."goods_services_id_seq"'::"regclass");
ALTER TABLE ONLY "public"."goods_services_comparisons" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."goods_services_comparisons_id_seq"'::"regclass");
ALTER TABLE ONLY "public"."mark_comparisons" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."mark_comparisons_id_seq"'::"regclass");
ALTER TABLE ONLY "public"."opponent_goods_services" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."opponent_goods_services_id_seq"'::"regclass");
ALTER TABLE ONLY "public"."opponent_marks" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."opponent_marks_id_seq"'::"regclass");
ALTER TABLE ONLY "public"."opposition_grounds" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."opposition_grounds_id_seq"'::"regclass");
ALTER TABLE ONLY "public"."precedents_cited" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."precedents_cited_id_seq"'::"regclass");
ALTER TABLE ONLY "public"."vector_embeddings" ALTER COLUMN "id" SET DEFAULT "nextval"('"public"."vector_embeddings_id_seq"'::"regclass");
ALTER TABLE ONLY "public"."applicant_goods_services"
    ADD CONSTRAINT "applicant_goods_services_mark_id_goods_services_id_key" UNIQUE ("mark_id", "goods_services_id");
ALTER TABLE ONLY "public"."applicant_goods_services"
    ADD CONSTRAINT "applicant_goods_services_pkey" PRIMARY KEY ("id");
ALTER TABLE ONLY "public"."applicant_marks"
    ADD CONSTRAINT "applicant_marks_pkey" PRIMARY KEY ("id");
ALTER TABLE ONLY "public"."decision_rationales"
    ADD CONSTRAINT "decision_rationales_pkey" PRIMARY KEY ("id");
ALTER TABLE ONLY "public"."goods_services_comparisons"
    ADD CONSTRAINT "goods_services_comparisons_goods_services_1_id_goods_servic_key" UNIQUE ("goods_services_1_id", "goods_services_2_id");
ALTER TABLE ONLY "public"."goods_services_comparisons"
    ADD CONSTRAINT "goods_services_comparisons_pkey" PRIMARY KEY ("id");
ALTER TABLE ONLY "public"."goods_services"
    ADD CONSTRAINT "goods_services_pkey" PRIMARY KEY ("id");
ALTER TABLE ONLY "public"."goods_services"
    ADD CONSTRAINT "goods_services_term_nice_class_key" UNIQUE ("term", "nice_class");
ALTER TABLE ONLY "public"."mark_comparisons"
    ADD CONSTRAINT "mark_comparisons_applicant_mark_id_opponent_mark_id_key" UNIQUE ("applicant_mark_id", "opponent_mark_id");
ALTER TABLE ONLY "public"."mark_comparisons"
    ADD CONSTRAINT "mark_comparisons_pkey" PRIMARY KEY ("id");
ALTER TABLE ONLY "public"."opponent_goods_services"
    ADD CONSTRAINT "opponent_goods_services_mark_id_goods_services_id_key" UNIQUE ("mark_id", "goods_services_id");
ALTER TABLE ONLY "public"."opponent_goods_services"
    ADD CONSTRAINT "opponent_goods_services_pkey" PRIMARY KEY ("id");
ALTER TABLE ONLY "public"."opponent_marks"
    ADD CONSTRAINT "opponent_marks_pkey" PRIMARY KEY ("id");
ALTER TABLE ONLY "public"."opposition_grounds"
    ADD CONSTRAINT "opposition_grounds_pkey" PRIMARY KEY ("id");
ALTER TABLE ONLY "public"."precedents_cited"
    ADD CONSTRAINT "precedents_cited_pkey" PRIMARY KEY ("id");
ALTER TABLE ONLY "public"."trademark_cases"
    ADD CONSTRAINT "trademark_cases_pkey" PRIMARY KEY ("case_reference");
ALTER TABLE ONLY "public"."vector_embeddings"
    ADD CONSTRAINT "vector_embeddings_pkey" PRIMARY KEY ("id");
CREATE INDEX "idx_applicant_marks_case" ON "public"."applicant_marks" USING "btree" ("case_reference");
CREATE INDEX "idx_case_reference" ON "public"."trademark_cases" USING "btree" ("case_reference");
CREATE INDEX "idx_goods_services_comparisons_market_context" ON "public"."goods_services_comparisons" USING "gin" ("market_context");
CREATE INDEX "idx_goods_services_comparisons_similarity" ON "public"."goods_services_comparisons" USING "btree" ("similarity");
CREATE INDEX "idx_goods_services_nice_class" ON "public"."goods_services" USING "btree" ("nice_class");
CREATE INDEX "idx_goods_services_term" ON "public"."goods_services" USING "btree" ("term");
CREATE INDEX "idx_mark_comparisons_case" ON "public"."mark_comparisons" USING "btree" ("case_reference");
CREATE INDEX "idx_opponent_marks_case" ON "public"."opponent_marks" USING "btree" ("case_reference");
CREATE INDEX "idx_trademark_cases_metadata" ON "public"."trademark_cases" USING "gin" ("metadata");
CREATE INDEX "idx_vector_embeddings_entity" ON "public"."vector_embeddings" USING "btree" ("entity_type", "entity_id");
CREATE OR REPLACE TRIGGER "update_applicant_goods_services_timestamp" BEFORE UPDATE ON "public"."applicant_goods_services" FOR EACH ROW EXECUTE FUNCTION "public"."update_timestamp"();
CREATE OR REPLACE TRIGGER "update_applicant_marks_timestamp" BEFORE UPDATE ON "public"."applicant_marks" FOR EACH ROW EXECUTE FUNCTION "public"."update_timestamp"();
CREATE OR REPLACE TRIGGER "update_decision_rationales_timestamp" BEFORE UPDATE ON "public"."decision_rationales" FOR EACH ROW EXECUTE FUNCTION "public"."update_timestamp"();
CREATE OR REPLACE TRIGGER "update_goods_services_comparisons_timestamp" BEFORE UPDATE ON "public"."goods_services_comparisons" FOR EACH ROW EXECUTE FUNCTION "public"."update_timestamp"();
CREATE OR REPLACE TRIGGER "update_goods_services_timestamp" BEFORE UPDATE ON "public"."goods_services" FOR EACH ROW EXECUTE FUNCTION "public"."update_timestamp"();
CREATE OR REPLACE TRIGGER "update_mark_comparisons_timestamp" BEFORE UPDATE ON "public"."mark_comparisons" FOR EACH ROW EXECUTE FUNCTION "public"."update_timestamp"();
CREATE OR REPLACE TRIGGER "update_opponent_goods_services_timestamp" BEFORE UPDATE ON "public"."opponent_goods_services" FOR EACH ROW EXECUTE FUNCTION "public"."update_timestamp"();
CREATE OR REPLACE TRIGGER "update_opponent_marks_timestamp" BEFORE UPDATE ON "public"."opponent_marks" FOR EACH ROW EXECUTE FUNCTION "public"."update_timestamp"();
CREATE OR REPLACE TRIGGER "update_opposition_grounds_timestamp" BEFORE UPDATE ON "public"."opposition_grounds" FOR EACH ROW EXECUTE FUNCTION "public"."update_timestamp"();
CREATE OR REPLACE TRIGGER "update_precedents_cited_timestamp" BEFORE UPDATE ON "public"."precedents_cited" FOR EACH ROW EXECUTE FUNCTION "public"."update_timestamp"();
CREATE OR REPLACE TRIGGER "update_trademark_cases_timestamp" BEFORE UPDATE ON "public"."trademark_cases" FOR EACH ROW EXECUTE FUNCTION "public"."update_timestamp"();
CREATE OR REPLACE TRIGGER "update_vector_embeddings_timestamp" BEFORE UPDATE ON "public"."vector_embeddings" FOR EACH ROW EXECUTE FUNCTION "public"."update_timestamp"();
ALTER TABLE ONLY "public"."applicant_goods_services"
    ADD CONSTRAINT "applicant_goods_services_goods_services_id_fkey" FOREIGN KEY ("goods_services_id") REFERENCES "public"."goods_services"("id") ON DELETE CASCADE;
ALTER TABLE ONLY "public"."applicant_goods_services"
    ADD CONSTRAINT "applicant_goods_services_mark_id_fkey" FOREIGN KEY ("mark_id") REFERENCES "public"."applicant_marks"("id") ON DELETE CASCADE;
ALTER TABLE ONLY "public"."applicant_marks"
    ADD CONSTRAINT "applicant_marks_case_reference_fkey" FOREIGN KEY ("case_reference") REFERENCES "public"."trademark_cases"("case_reference") ON DELETE CASCADE;
ALTER TABLE ONLY "public"."decision_rationales"
    ADD CONSTRAINT "decision_rationales_case_reference_fkey" FOREIGN KEY ("case_reference") REFERENCES "public"."trademark_cases"("case_reference") ON DELETE CASCADE;
ALTER TABLE ONLY "public"."goods_services_comparisons"
    ADD CONSTRAINT "goods_services_comparisons_case_reference_fkey" FOREIGN KEY ("case_reference") REFERENCES "public"."trademark_cases"("case_reference") ON DELETE SET NULL;
ALTER TABLE ONLY "public"."goods_services_comparisons"
    ADD CONSTRAINT "goods_services_comparisons_goods_services_1_id_fkey" FOREIGN KEY ("goods_services_1_id") REFERENCES "public"."goods_services"("id") ON DELETE CASCADE;
ALTER TABLE ONLY "public"."goods_services_comparisons"
    ADD CONSTRAINT "goods_services_comparisons_goods_services_2_id_fkey" FOREIGN KEY ("goods_services_2_id") REFERENCES "public"."goods_services"("id") ON DELETE CASCADE;
ALTER TABLE ONLY "public"."mark_comparisons"
    ADD CONSTRAINT "mark_comparisons_applicant_mark_id_fkey" FOREIGN KEY ("applicant_mark_id") REFERENCES "public"."applicant_marks"("id") ON DELETE CASCADE;
ALTER TABLE ONLY "public"."mark_comparisons"
    ADD CONSTRAINT "mark_comparisons_case_reference_fkey" FOREIGN KEY ("case_reference") REFERENCES "public"."trademark_cases"("case_reference") ON DELETE CASCADE;
ALTER TABLE ONLY "public"."mark_comparisons"
    ADD CONSTRAINT "mark_comparisons_opponent_mark_id_fkey" FOREIGN KEY ("opponent_mark_id") REFERENCES "public"."opponent_marks"("id") ON DELETE CASCADE;
ALTER TABLE ONLY "public"."opponent_goods_services"
    ADD CONSTRAINT "opponent_goods_services_goods_services_id_fkey" FOREIGN KEY ("goods_services_id") REFERENCES "public"."goods_services"("id") ON DELETE CASCADE;
ALTER TABLE ONLY "public"."opponent_goods_services"
    ADD CONSTRAINT "opponent_goods_services_mark_id_fkey" FOREIGN KEY ("mark_id") REFERENCES "public"."opponent_marks"("id") ON DELETE CASCADE;
ALTER TABLE ONLY "public"."opponent_marks"
    ADD CONSTRAINT "opponent_marks_case_reference_fkey" FOREIGN KEY ("case_reference") REFERENCES "public"."trademark_cases"("case_reference") ON DELETE CASCADE;
ALTER TABLE ONLY "public"."opposition_grounds"
    ADD CONSTRAINT "opposition_grounds_case_reference_fkey" FOREIGN KEY ("case_reference") REFERENCES "public"."trademark_cases"("case_reference") ON DELETE CASCADE;
ALTER TABLE ONLY "public"."precedents_cited"
    ADD CONSTRAINT "precedents_cited_decision_rationale_id_fkey" FOREIGN KEY ("decision_rationale_id") REFERENCES "public"."decision_rationales"("id") ON DELETE CASCADE;
ALTER PUBLICATION "supabase_realtime" OWNER TO "postgres";
GRANT USAGE ON SCHEMA "public" TO "postgres";
GRANT USAGE ON SCHEMA "public" TO "anon";
GRANT USAGE ON SCHEMA "public" TO "authenticated";
GRANT USAGE ON SCHEMA "public" TO "service_role";
GRANT ALL ON FUNCTION "public"."gtrgm_in"("cstring") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_in"("cstring") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_in"("cstring") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_in"("cstring") TO "service_role";
GRANT ALL ON FUNCTION "public"."gtrgm_out"("public"."gtrgm") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_out"("public"."gtrgm") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_out"("public"."gtrgm") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_out"("public"."gtrgm") TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_in"("cstring", "oid", integer) TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_in"("cstring", "oid", integer) TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_in"("cstring", "oid", integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_in"("cstring", "oid", integer) TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_out"("public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_out"("public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_out"("public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_out"("public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_recv"("internal", "oid", integer) TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_recv"("internal", "oid", integer) TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_recv"("internal", "oid", integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_recv"("internal", "oid", integer) TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_send"("public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_send"("public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_send"("public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_send"("public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_typmod_in"("cstring"[]) TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_typmod_in"("cstring"[]) TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_typmod_in"("cstring"[]) TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_typmod_in"("cstring"[]) TO "service_role";
GRANT ALL ON FUNCTION "public"."sparsevec_in"("cstring", "oid", integer) TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_in"("cstring", "oid", integer) TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_in"("cstring", "oid", integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_in"("cstring", "oid", integer) TO "service_role";
GRANT ALL ON FUNCTION "public"."sparsevec_out"("public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_out"("public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_out"("public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_out"("public"."sparsevec") TO "service_role";
GRANT ALL ON FUNCTION "public"."sparsevec_recv"("internal", "oid", integer) TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_recv"("internal", "oid", integer) TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_recv"("internal", "oid", integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_recv"("internal", "oid", integer) TO "service_role";
GRANT ALL ON FUNCTION "public"."sparsevec_send"("public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_send"("public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_send"("public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_send"("public"."sparsevec") TO "service_role";
GRANT ALL ON FUNCTION "public"."sparsevec_typmod_in"("cstring"[]) TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_typmod_in"("cstring"[]) TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_typmod_in"("cstring"[]) TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_typmod_in"("cstring"[]) TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_in"("cstring", "oid", integer) TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_in"("cstring", "oid", integer) TO "anon";
GRANT ALL ON FUNCTION "public"."vector_in"("cstring", "oid", integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_in"("cstring", "oid", integer) TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_out"("public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_out"("public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_out"("public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_out"("public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_recv"("internal", "oid", integer) TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_recv"("internal", "oid", integer) TO "anon";
GRANT ALL ON FUNCTION "public"."vector_recv"("internal", "oid", integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_recv"("internal", "oid", integer) TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_send"("public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_send"("public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_send"("public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_send"("public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_typmod_in"("cstring"[]) TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_typmod_in"("cstring"[]) TO "anon";
GRANT ALL ON FUNCTION "public"."vector_typmod_in"("cstring"[]) TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_typmod_in"("cstring"[]) TO "service_role";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(real[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(real[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(real[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(real[], integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(real[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(real[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(real[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(real[], integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."array_to_vector"(real[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_vector"(real[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_vector"(real[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_vector"(real[], integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(double precision[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(double precision[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(double precision[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(double precision[], integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(double precision[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(double precision[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(double precision[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(double precision[], integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."array_to_vector"(double precision[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_vector"(double precision[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_vector"(double precision[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_vector"(double precision[], integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(integer[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(integer[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(integer[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(integer[], integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(integer[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(integer[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(integer[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(integer[], integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."array_to_vector"(integer[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_vector"(integer[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_vector"(integer[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_vector"(integer[], integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(numeric[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(numeric[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(numeric[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_halfvec"(numeric[], integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(numeric[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(numeric[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(numeric[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_sparsevec"(numeric[], integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."array_to_vector"(numeric[], integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."array_to_vector"(numeric[], integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."array_to_vector"(numeric[], integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."array_to_vector"(numeric[], integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_to_float4"("public"."halfvec", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_to_float4"("public"."halfvec", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_to_float4"("public"."halfvec", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_to_float4"("public"."halfvec", integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec"("public"."halfvec", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec"("public"."halfvec", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec"("public"."halfvec", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec"("public"."halfvec", integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_to_sparsevec"("public"."halfvec", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_to_sparsevec"("public"."halfvec", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_to_sparsevec"("public"."halfvec", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_to_sparsevec"("public"."halfvec", integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_to_vector"("public"."halfvec", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_to_vector"("public"."halfvec", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_to_vector"("public"."halfvec", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_to_vector"("public"."halfvec", integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."sparsevec_to_halfvec"("public"."sparsevec", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_to_halfvec"("public"."sparsevec", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_to_halfvec"("public"."sparsevec", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_to_halfvec"("public"."sparsevec", integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."sparsevec"("public"."sparsevec", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec"("public"."sparsevec", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec"("public"."sparsevec", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec"("public"."sparsevec", integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."sparsevec_to_vector"("public"."sparsevec", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_to_vector"("public"."sparsevec", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_to_vector"("public"."sparsevec", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_to_vector"("public"."sparsevec", integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_to_float4"("public"."vector", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_to_float4"("public"."vector", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."vector_to_float4"("public"."vector", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_to_float4"("public"."vector", integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_to_halfvec"("public"."vector", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_to_halfvec"("public"."vector", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."vector_to_halfvec"("public"."vector", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_to_halfvec"("public"."vector", integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_to_sparsevec"("public"."vector", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_to_sparsevec"("public"."vector", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."vector_to_sparsevec"("public"."vector", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_to_sparsevec"("public"."vector", integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."vector"("public"."vector", integer, boolean) TO "postgres";
GRANT ALL ON FUNCTION "public"."vector"("public"."vector", integer, boolean) TO "anon";
GRANT ALL ON FUNCTION "public"."vector"("public"."vector", integer, boolean) TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector"("public"."vector", integer, boolean) TO "service_role";
GRANT ALL ON FUNCTION "public"."begin_transaction"() TO "anon";
GRANT ALL ON FUNCTION "public"."begin_transaction"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."begin_transaction"() TO "service_role";
GRANT ALL ON FUNCTION "public"."binary_quantize"("public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."binary_quantize"("public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."binary_quantize"("public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."binary_quantize"("public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."binary_quantize"("public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."binary_quantize"("public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."binary_quantize"("public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."binary_quantize"("public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."commit_transaction"() TO "anon";
GRANT ALL ON FUNCTION "public"."commit_transaction"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."commit_transaction"() TO "service_role";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."halfvec", "public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."sparsevec", "public"."sparsevec") TO "service_role";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."cosine_distance"("public"."vector", "public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."find_similar_cases"("mark_text" "text", "limit_count" integer) TO "anon";
GRANT ALL ON FUNCTION "public"."find_similar_cases"("mark_text" "text", "limit_count" integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."find_similar_cases"("mark_text" "text", "limit_count" integer) TO "service_role";
GRANT ALL ON FUNCTION "public"."find_similar_goods_services"("search_term" "text", "class_num" integer, "limit_count" integer) TO "anon";
GRANT ALL ON FUNCTION "public"."find_similar_goods_services"("search_term" "text", "class_num" integer, "limit_count" integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."find_similar_goods_services"("search_term" "text", "class_num" integer, "limit_count" integer) TO "service_role";
GRANT ALL ON FUNCTION "public"."find_similar_term"("input_term" "text", "input_nice_class" integer, "threshold" double precision) TO "anon";
GRANT ALL ON FUNCTION "public"."find_similar_term"("input_term" "text", "input_nice_class" integer, "threshold" double precision) TO "authenticated";
GRANT ALL ON FUNCTION "public"."find_similar_term"("input_term" "text", "input_nice_class" integer, "threshold" double precision) TO "service_role";
GRANT ALL ON FUNCTION "public"."get_goods_services_comparisons"("limit_count" integer) TO "anon";
GRANT ALL ON FUNCTION "public"."get_goods_services_comparisons"("limit_count" integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."get_goods_services_comparisons"("limit_count" integer) TO "service_role";
GRANT ALL ON FUNCTION "public"."gin_extract_query_trgm"("text", "internal", smallint, "internal", "internal", "internal", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gin_extract_query_trgm"("text", "internal", smallint, "internal", "internal", "internal", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gin_extract_query_trgm"("text", "internal", smallint, "internal", "internal", "internal", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gin_extract_query_trgm"("text", "internal", smallint, "internal", "internal", "internal", "internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."gin_extract_value_trgm"("text", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gin_extract_value_trgm"("text", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gin_extract_value_trgm"("text", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gin_extract_value_trgm"("text", "internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."gin_trgm_consistent"("internal", smallint, "text", integer, "internal", "internal", "internal", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gin_trgm_consistent"("internal", smallint, "text", integer, "internal", "internal", "internal", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gin_trgm_consistent"("internal", smallint, "text", integer, "internal", "internal", "internal", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gin_trgm_consistent"("internal", smallint, "text", integer, "internal", "internal", "internal", "internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."gin_trgm_triconsistent"("internal", smallint, "text", integer, "internal", "internal", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gin_trgm_triconsistent"("internal", smallint, "text", integer, "internal", "internal", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gin_trgm_triconsistent"("internal", smallint, "text", integer, "internal", "internal", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gin_trgm_triconsistent"("internal", smallint, "text", integer, "internal", "internal", "internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."gtrgm_compress"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_compress"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_compress"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_compress"("internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."gtrgm_consistent"("internal", "text", smallint, "oid", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_consistent"("internal", "text", smallint, "oid", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_consistent"("internal", "text", smallint, "oid", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_consistent"("internal", "text", smallint, "oid", "internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."gtrgm_decompress"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_decompress"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_decompress"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_decompress"("internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."gtrgm_distance"("internal", "text", smallint, "oid", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_distance"("internal", "text", smallint, "oid", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_distance"("internal", "text", smallint, "oid", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_distance"("internal", "text", smallint, "oid", "internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."gtrgm_options"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_options"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_options"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_options"("internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."gtrgm_penalty"("internal", "internal", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_penalty"("internal", "internal", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_penalty"("internal", "internal", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_penalty"("internal", "internal", "internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."gtrgm_picksplit"("internal", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_picksplit"("internal", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_picksplit"("internal", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_picksplit"("internal", "internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."gtrgm_same"("public"."gtrgm", "public"."gtrgm", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_same"("public"."gtrgm", "public"."gtrgm", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_same"("public"."gtrgm", "public"."gtrgm", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_same"("public"."gtrgm", "public"."gtrgm", "internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."gtrgm_union"("internal", "internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."gtrgm_union"("internal", "internal") TO "anon";
GRANT ALL ON FUNCTION "public"."gtrgm_union"("internal", "internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."gtrgm_union"("internal", "internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_accum"(double precision[], "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_accum"(double precision[], "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_accum"(double precision[], "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_accum"(double precision[], "public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_add"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_add"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_add"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_add"("public"."halfvec", "public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_avg"(double precision[]) TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_avg"(double precision[]) TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_avg"(double precision[]) TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_avg"(double precision[]) TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_cmp"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_cmp"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_cmp"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_cmp"("public"."halfvec", "public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_combine"(double precision[], double precision[]) TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_combine"(double precision[], double precision[]) TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_combine"(double precision[], double precision[]) TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_combine"(double precision[], double precision[]) TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_concat"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_concat"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_concat"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_concat"("public"."halfvec", "public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_eq"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_eq"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_eq"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_eq"("public"."halfvec", "public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_ge"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_ge"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_ge"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_ge"("public"."halfvec", "public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_gt"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_gt"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_gt"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_gt"("public"."halfvec", "public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_l2_squared_distance"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_l2_squared_distance"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_l2_squared_distance"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_l2_squared_distance"("public"."halfvec", "public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_le"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_le"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_le"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_le"("public"."halfvec", "public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_lt"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_lt"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_lt"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_lt"("public"."halfvec", "public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_mul"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_mul"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_mul"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_mul"("public"."halfvec", "public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_ne"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_ne"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_ne"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_ne"("public"."halfvec", "public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_negative_inner_product"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_negative_inner_product"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_negative_inner_product"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_negative_inner_product"("public"."halfvec", "public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_spherical_distance"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_spherical_distance"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_spherical_distance"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_spherical_distance"("public"."halfvec", "public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."halfvec_sub"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."halfvec_sub"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."halfvec_sub"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."halfvec_sub"("public"."halfvec", "public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."hamming_distance"(bit, bit) TO "postgres";
GRANT ALL ON FUNCTION "public"."hamming_distance"(bit, bit) TO "anon";
GRANT ALL ON FUNCTION "public"."hamming_distance"(bit, bit) TO "authenticated";
GRANT ALL ON FUNCTION "public"."hamming_distance"(bit, bit) TO "service_role";
GRANT ALL ON FUNCTION "public"."hnsw_bit_support"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."hnsw_bit_support"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."hnsw_bit_support"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."hnsw_bit_support"("internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."hnsw_halfvec_support"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."hnsw_halfvec_support"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."hnsw_halfvec_support"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."hnsw_halfvec_support"("internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."hnsw_sparsevec_support"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."hnsw_sparsevec_support"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."hnsw_sparsevec_support"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."hnsw_sparsevec_support"("internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."hnswhandler"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."hnswhandler"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."hnswhandler"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."hnswhandler"("internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."halfvec", "public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."sparsevec", "public"."sparsevec") TO "service_role";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."inner_product"("public"."vector", "public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."ivfflat_bit_support"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."ivfflat_bit_support"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."ivfflat_bit_support"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."ivfflat_bit_support"("internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."ivfflat_halfvec_support"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."ivfflat_halfvec_support"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."ivfflat_halfvec_support"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."ivfflat_halfvec_support"("internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."ivfflathandler"("internal") TO "postgres";
GRANT ALL ON FUNCTION "public"."ivfflathandler"("internal") TO "anon";
GRANT ALL ON FUNCTION "public"."ivfflathandler"("internal") TO "authenticated";
GRANT ALL ON FUNCTION "public"."ivfflathandler"("internal") TO "service_role";
GRANT ALL ON FUNCTION "public"."jaccard_distance"(bit, bit) TO "postgres";
GRANT ALL ON FUNCTION "public"."jaccard_distance"(bit, bit) TO "anon";
GRANT ALL ON FUNCTION "public"."jaccard_distance"(bit, bit) TO "authenticated";
GRANT ALL ON FUNCTION "public"."jaccard_distance"(bit, bit) TO "service_role";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."halfvec", "public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."sparsevec", "public"."sparsevec") TO "service_role";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l1_distance"("public"."vector", "public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."halfvec", "public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."halfvec", "public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."halfvec", "public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."halfvec", "public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."sparsevec", "public"."sparsevec") TO "service_role";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l2_distance"("public"."vector", "public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."l2_norm"("public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."l2_norm"("public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."l2_norm"("public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l2_norm"("public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."l2_norm"("public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."l2_norm"("public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."l2_norm"("public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l2_norm"("public"."sparsevec") TO "service_role";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."sparsevec") TO "service_role";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."l2_normalize"("public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."predict_similarity"("term1" "text", "nice_class1" integer, "term2" "text", "nice_class2" integer) TO "anon";
GRANT ALL ON FUNCTION "public"."predict_similarity"("term1" "text", "nice_class1" integer, "term2" "text", "nice_class2" integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."predict_similarity"("term1" "text", "nice_class1" integer, "term2" "text", "nice_class2" integer) TO "service_role";
GRANT ALL ON FUNCTION "public"."rollback_transaction"() TO "anon";
GRANT ALL ON FUNCTION "public"."rollback_transaction"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."rollback_transaction"() TO "service_role";
GRANT ALL ON FUNCTION "public"."set_limit"(real) TO "postgres";
GRANT ALL ON FUNCTION "public"."set_limit"(real) TO "anon";
GRANT ALL ON FUNCTION "public"."set_limit"(real) TO "authenticated";
GRANT ALL ON FUNCTION "public"."set_limit"(real) TO "service_role";
GRANT ALL ON FUNCTION "public"."show_limit"() TO "postgres";
GRANT ALL ON FUNCTION "public"."show_limit"() TO "anon";
GRANT ALL ON FUNCTION "public"."show_limit"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."show_limit"() TO "service_role";
GRANT ALL ON FUNCTION "public"."show_trgm"("text") TO "postgres";
GRANT ALL ON FUNCTION "public"."show_trgm"("text") TO "anon";
GRANT ALL ON FUNCTION "public"."show_trgm"("text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."show_trgm"("text") TO "service_role";
GRANT ALL ON FUNCTION "public"."similarity"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."similarity"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."similarity"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."similarity"("text", "text") TO "service_role";
GRANT ALL ON FUNCTION "public"."similarity_dist"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."similarity_dist"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."similarity_dist"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."similarity_dist"("text", "text") TO "service_role";
GRANT ALL ON FUNCTION "public"."similarity_op"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."similarity_op"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."similarity_op"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."similarity_op"("text", "text") TO "service_role";
GRANT ALL ON FUNCTION "public"."sparsevec_cmp"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_cmp"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_cmp"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_cmp"("public"."sparsevec", "public"."sparsevec") TO "service_role";
GRANT ALL ON FUNCTION "public"."sparsevec_eq"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_eq"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_eq"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_eq"("public"."sparsevec", "public"."sparsevec") TO "service_role";
GRANT ALL ON FUNCTION "public"."sparsevec_ge"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_ge"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_ge"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_ge"("public"."sparsevec", "public"."sparsevec") TO "service_role";
GRANT ALL ON FUNCTION "public"."sparsevec_gt"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_gt"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_gt"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_gt"("public"."sparsevec", "public"."sparsevec") TO "service_role";
GRANT ALL ON FUNCTION "public"."sparsevec_l2_squared_distance"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_l2_squared_distance"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_l2_squared_distance"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_l2_squared_distance"("public"."sparsevec", "public"."sparsevec") TO "service_role";
GRANT ALL ON FUNCTION "public"."sparsevec_le"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_le"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_le"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_le"("public"."sparsevec", "public"."sparsevec") TO "service_role";
GRANT ALL ON FUNCTION "public"."sparsevec_lt"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_lt"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_lt"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_lt"("public"."sparsevec", "public"."sparsevec") TO "service_role";
GRANT ALL ON FUNCTION "public"."sparsevec_ne"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_ne"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_ne"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_ne"("public"."sparsevec", "public"."sparsevec") TO "service_role";
GRANT ALL ON FUNCTION "public"."sparsevec_negative_inner_product"("public"."sparsevec", "public"."sparsevec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sparsevec_negative_inner_product"("public"."sparsevec", "public"."sparsevec") TO "anon";
GRANT ALL ON FUNCTION "public"."sparsevec_negative_inner_product"("public"."sparsevec", "public"."sparsevec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sparsevec_negative_inner_product"("public"."sparsevec", "public"."sparsevec") TO "service_role";
GRANT ALL ON FUNCTION "public"."strict_word_similarity"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."strict_word_similarity"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."strict_word_similarity"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."strict_word_similarity"("text", "text") TO "service_role";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_commutator_op"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_commutator_op"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_commutator_op"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_commutator_op"("text", "text") TO "service_role";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_dist_commutator_op"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_dist_commutator_op"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_dist_commutator_op"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_dist_commutator_op"("text", "text") TO "service_role";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_dist_op"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_dist_op"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_dist_op"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_dist_op"("text", "text") TO "service_role";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_op"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_op"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_op"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."strict_word_similarity_op"("text", "text") TO "service_role";
GRANT ALL ON FUNCTION "public"."subvector"("public"."halfvec", integer, integer) TO "postgres";
GRANT ALL ON FUNCTION "public"."subvector"("public"."halfvec", integer, integer) TO "anon";
GRANT ALL ON FUNCTION "public"."subvector"("public"."halfvec", integer, integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."subvector"("public"."halfvec", integer, integer) TO "service_role";
GRANT ALL ON FUNCTION "public"."subvector"("public"."vector", integer, integer) TO "postgres";
GRANT ALL ON FUNCTION "public"."subvector"("public"."vector", integer, integer) TO "anon";
GRANT ALL ON FUNCTION "public"."subvector"("public"."vector", integer, integer) TO "authenticated";
GRANT ALL ON FUNCTION "public"."subvector"("public"."vector", integer, integer) TO "service_role";
GRANT ALL ON FUNCTION "public"."update_timestamp"() TO "anon";
GRANT ALL ON FUNCTION "public"."update_timestamp"() TO "authenticated";
GRANT ALL ON FUNCTION "public"."update_timestamp"() TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_accum"(double precision[], "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_accum"(double precision[], "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_accum"(double precision[], "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_accum"(double precision[], "public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_add"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_add"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_add"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_add"("public"."vector", "public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_avg"(double precision[]) TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_avg"(double precision[]) TO "anon";
GRANT ALL ON FUNCTION "public"."vector_avg"(double precision[]) TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_avg"(double precision[]) TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_cmp"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_cmp"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_cmp"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_cmp"("public"."vector", "public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_combine"(double precision[], double precision[]) TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_combine"(double precision[], double precision[]) TO "anon";
GRANT ALL ON FUNCTION "public"."vector_combine"(double precision[], double precision[]) TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_combine"(double precision[], double precision[]) TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_concat"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_concat"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_concat"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_concat"("public"."vector", "public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_dims"("public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_dims"("public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_dims"("public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_dims"("public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_dims"("public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_dims"("public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_dims"("public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_dims"("public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_eq"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_eq"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_eq"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_eq"("public"."vector", "public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_ge"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_ge"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_ge"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_ge"("public"."vector", "public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_gt"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_gt"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_gt"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_gt"("public"."vector", "public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_l2_squared_distance"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_l2_squared_distance"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_l2_squared_distance"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_l2_squared_distance"("public"."vector", "public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_le"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_le"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_le"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_le"("public"."vector", "public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_lt"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_lt"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_lt"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_lt"("public"."vector", "public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_mul"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_mul"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_mul"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_mul"("public"."vector", "public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_ne"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_ne"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_ne"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_ne"("public"."vector", "public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_negative_inner_product"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_negative_inner_product"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_negative_inner_product"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_negative_inner_product"("public"."vector", "public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_norm"("public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_norm"("public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_norm"("public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_norm"("public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_spherical_distance"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_spherical_distance"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_spherical_distance"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_spherical_distance"("public"."vector", "public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."vector_sub"("public"."vector", "public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."vector_sub"("public"."vector", "public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."vector_sub"("public"."vector", "public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."vector_sub"("public"."vector", "public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."word_similarity"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."word_similarity"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."word_similarity"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."word_similarity"("text", "text") TO "service_role";
GRANT ALL ON FUNCTION "public"."word_similarity_commutator_op"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."word_similarity_commutator_op"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."word_similarity_commutator_op"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."word_similarity_commutator_op"("text", "text") TO "service_role";
GRANT ALL ON FUNCTION "public"."word_similarity_dist_commutator_op"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."word_similarity_dist_commutator_op"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."word_similarity_dist_commutator_op"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."word_similarity_dist_commutator_op"("text", "text") TO "service_role";
GRANT ALL ON FUNCTION "public"."word_similarity_dist_op"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."word_similarity_dist_op"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."word_similarity_dist_op"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."word_similarity_dist_op"("text", "text") TO "service_role";
GRANT ALL ON FUNCTION "public"."word_similarity_op"("text", "text") TO "postgres";
GRANT ALL ON FUNCTION "public"."word_similarity_op"("text", "text") TO "anon";
GRANT ALL ON FUNCTION "public"."word_similarity_op"("text", "text") TO "authenticated";
GRANT ALL ON FUNCTION "public"."word_similarity_op"("text", "text") TO "service_role";
GRANT ALL ON FUNCTION "public"."avg"("public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."avg"("public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."avg"("public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."avg"("public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."avg"("public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."avg"("public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."avg"("public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."avg"("public"."vector") TO "service_role";
GRANT ALL ON FUNCTION "public"."sum"("public"."halfvec") TO "postgres";
GRANT ALL ON FUNCTION "public"."sum"("public"."halfvec") TO "anon";
GRANT ALL ON FUNCTION "public"."sum"("public"."halfvec") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sum"("public"."halfvec") TO "service_role";
GRANT ALL ON FUNCTION "public"."sum"("public"."vector") TO "postgres";
GRANT ALL ON FUNCTION "public"."sum"("public"."vector") TO "anon";
GRANT ALL ON FUNCTION "public"."sum"("public"."vector") TO "authenticated";
GRANT ALL ON FUNCTION "public"."sum"("public"."vector") TO "service_role";
GRANT ALL ON TABLE "public"."applicant_goods_services" TO "anon";
GRANT ALL ON TABLE "public"."applicant_goods_services" TO "authenticated";
GRANT ALL ON TABLE "public"."applicant_goods_services" TO "service_role";
GRANT ALL ON SEQUENCE "public"."applicant_goods_services_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."applicant_goods_services_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."applicant_goods_services_id_seq" TO "service_role";
GRANT ALL ON TABLE "public"."applicant_marks" TO "anon";
GRANT ALL ON TABLE "public"."applicant_marks" TO "authenticated";
GRANT ALL ON TABLE "public"."applicant_marks" TO "service_role";
GRANT ALL ON SEQUENCE "public"."applicant_marks_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."applicant_marks_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."applicant_marks_id_seq" TO "service_role";
GRANT ALL ON TABLE "public"."decision_rationales" TO "anon";
GRANT ALL ON TABLE "public"."decision_rationales" TO "authenticated";
GRANT ALL ON TABLE "public"."decision_rationales" TO "service_role";
GRANT ALL ON SEQUENCE "public"."decision_rationales_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."decision_rationales_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."decision_rationales_id_seq" TO "service_role";
GRANT ALL ON TABLE "public"."goods_services" TO "anon";
GRANT ALL ON TABLE "public"."goods_services" TO "authenticated";
GRANT ALL ON TABLE "public"."goods_services" TO "service_role";
GRANT ALL ON TABLE "public"."goods_services_comparisons" TO "anon";
GRANT ALL ON TABLE "public"."goods_services_comparisons" TO "authenticated";
GRANT ALL ON TABLE "public"."goods_services_comparisons" TO "service_role";
GRANT ALL ON SEQUENCE "public"."goods_services_comparisons_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."goods_services_comparisons_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."goods_services_comparisons_id_seq" TO "service_role";
GRANT ALL ON SEQUENCE "public"."goods_services_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."goods_services_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."goods_services_id_seq" TO "service_role";
GRANT ALL ON TABLE "public"."mark_comparisons" TO "anon";
GRANT ALL ON TABLE "public"."mark_comparisons" TO "authenticated";
GRANT ALL ON TABLE "public"."mark_comparisons" TO "service_role";
GRANT ALL ON SEQUENCE "public"."mark_comparisons_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."mark_comparisons_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."mark_comparisons_id_seq" TO "service_role";
GRANT ALL ON TABLE "public"."opponent_goods_services" TO "anon";
GRANT ALL ON TABLE "public"."opponent_goods_services" TO "authenticated";
GRANT ALL ON TABLE "public"."opponent_goods_services" TO "service_role";
GRANT ALL ON SEQUENCE "public"."opponent_goods_services_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."opponent_goods_services_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."opponent_goods_services_id_seq" TO "service_role";
GRANT ALL ON TABLE "public"."opponent_marks" TO "anon";
GRANT ALL ON TABLE "public"."opponent_marks" TO "authenticated";
GRANT ALL ON TABLE "public"."opponent_marks" TO "service_role";
GRANT ALL ON SEQUENCE "public"."opponent_marks_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."opponent_marks_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."opponent_marks_id_seq" TO "service_role";
GRANT ALL ON TABLE "public"."opposition_grounds" TO "anon";
GRANT ALL ON TABLE "public"."opposition_grounds" TO "authenticated";
GRANT ALL ON TABLE "public"."opposition_grounds" TO "service_role";
GRANT ALL ON SEQUENCE "public"."opposition_grounds_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."opposition_grounds_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."opposition_grounds_id_seq" TO "service_role";
GRANT ALL ON TABLE "public"."precedents_cited" TO "anon";
GRANT ALL ON TABLE "public"."precedents_cited" TO "authenticated";
GRANT ALL ON TABLE "public"."precedents_cited" TO "service_role";
GRANT ALL ON SEQUENCE "public"."precedents_cited_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."precedents_cited_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."precedents_cited_id_seq" TO "service_role";
GRANT ALL ON TABLE "public"."trademark_cases" TO "anon";
GRANT ALL ON TABLE "public"."trademark_cases" TO "authenticated";
GRANT ALL ON TABLE "public"."trademark_cases" TO "service_role";
GRANT ALL ON TABLE "public"."vector_embeddings" TO "anon";
GRANT ALL ON TABLE "public"."vector_embeddings" TO "authenticated";
GRANT ALL ON TABLE "public"."vector_embeddings" TO "service_role";
GRANT ALL ON SEQUENCE "public"."vector_embeddings_id_seq" TO "anon";
GRANT ALL ON SEQUENCE "public"."vector_embeddings_id_seq" TO "authenticated";
GRANT ALL ON SEQUENCE "public"."vector_embeddings_id_seq" TO "service_role";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON SEQUENCES  TO "service_role";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON FUNCTIONS  TO "service_role";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "postgres";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "anon";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "authenticated";
ALTER DEFAULT PRIVILEGES FOR ROLE "postgres" IN SCHEMA "public" GRANT ALL ON TABLES  TO "service_role";
RESET ALL;
