import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
pc = Pinecone(api_key="pcsk_2Fu9XY_M3GbSvsnkHu2gRK45pTUExpJ6Se2mM8zFEecLsxdef5tb1W3N2f1H9ofTu2oxSc")
MEDICAL_INDEX = 'medical-literature'
EMBED_DIM = 1536

existing = [idx.name for idx in pc.list_indexes()]
if MEDICAL_INDEX not in existing:
    print(f'Creating index: {MEDICAL_INDEX}...')
    pc.create_index(
        name=MEDICAL_INDEX,
        dimension=EMBED_DIM,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1',
        ),
    )
    print(f'Index created: {MEDICAL_INDEX}')
else:
    print(f'Index already exists: {MEDICAL_INDEX}')

index = pc.Index(MEDICAL_INDEX)
stats = index.describe_index_stats()
print(f'{MEDICAL_INDEX}: {stats.total_vector_count} vectors, dimension={stats.dimension}')