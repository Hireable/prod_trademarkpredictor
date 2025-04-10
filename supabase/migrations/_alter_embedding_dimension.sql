ALTER TABLE public.vector_embeddings ALTER COLUMN embedding TYPE vector(384) USING embedding::vector(384);
