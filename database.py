from urllib.parse import quote
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

user = "root"
pwd = "1111"
host = "localhost"
port = 3306
db='mini'
db_url = f'mysql+pymysql://{user}:{quote(pwd)}@{host}:{port}/{db}'
engine = create_engine(db_url,echo=True)
SessionFactory = sessionmaker(autocommit=False,autoflush=False,bind=engine)

def get_db():
    session=SessionFactory()
    try:
        yield session
    finally:
        session.close()