gcloud run deploy trademark-agent \
    --source=. \
    --region=europe-west2 \
    --allow-unauthenticated \
    --env-vars-file=.env.yaml \
    --memory=1Gi \
    --cpu=3