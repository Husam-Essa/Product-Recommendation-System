from urllib.parse import quote_plus

username = "root"
password = quote_plus("")
host = "localhost"
database = "ec_db"

DATABASE_URL = f"mysql+mysqlconnector://{username}:{password}@{host}/{database}"
