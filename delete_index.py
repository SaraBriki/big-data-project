
import pinecone
import os
import dotenv
dotenv.load_dotenv()
index_name='image-search'
pinecone.init(
    api_key=os.environ.get('API_KEY'),
    environment="us-east-1-aws"  # find next to API key in console
)
pinecone.delete_index(index_name)