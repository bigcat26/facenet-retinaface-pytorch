import json
import numpy as np
from sqlalchemy import Column, String, Integer, LargeBinary, TypeDecorator, type_coerce, inspect, create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.dialects.postgresql import BYTEA

class Base:
    __allow_unmapped__ = True

Base = declarative_base(cls=Base)

class NumpyArray(TypeDecorator):
    impl = BYTEA
    cache_ok = True

    def __init__(self):
        super(NumpyArray, self).__init__()

    def process_bind_param(self, value, dialect):
        return value.tobytes()

    def process_result_value(self, value, dialect):
        return np.frombuffer(value)

class FaceFeature(Base):
    __tablename__ = "feature"

    pid = Column(Integer, primary_key=True)
    feat = Column(NumpyArray())

    def __init__(self, pid, feat):
        self.pid = pid
        self.feat = feat

class FacePerson(Base):
    __tablename__ = "person"

    pid = Column(Integer, primary_key=True, unique=True)
    name = Column(String(64))

    def __init__(self, pid, name):
        self.pid = pid
        self.name = name

class FaceDatabase:
    engine = None
    db_session = None

    def __init__(self) -> None:
        pass

    def open(self, **kwargs):
        # with open(file, 'rb') as f:
        self.engine = create_engine(f'postgresql://{kwargs["db_user"]}:{kwargs["db_passwd"]}@{kwargs["db_host"]}:{kwargs["db_port"]}/{kwargs["db_name"]}')
        self.db_session = sessionmaker(bind=self.engine)

    def close(self):
        if self.engine is not None:
            self.engine.dispose()

    def register(self, name: String, feat):
        person = FacePerson(None, name)
        feature = FaceFeature(None, feat)
        with self.db_session() as session:
            session.add(person)
            session.commit()
            feature.pid = person.pid
            session.add(feature)
            session.commit()
            return person.pid

    def unregister(self, pid: int):
        with self.db_session() as session:
            user = session.query(FacePerson).filter(FacePerson.pid == pid).first()
            if user:
                session.query(FaceFeature).filter(FaceFeature.pid == pid).delete()
                session.delete(user)
                session.commit()
                return True
        return False

    # def unregister_by_name(self, name: String):
    #     with self.db_session() as session:
    #         session.query(FacePerson).filter(FacePerson.name == name).delete()
    #         session.commit()

    def features(self):
        with self.db_session() as session:
            feats = session.query(FaceFeature).all()
            return feats

if __name__ == "__main__":
    with open('credentials.json', 'r') as f:
        cfg = json.load(f)
    # engine = create_engine(f'postgresql://{cfg["db_user"]}:{cfg["db_passwd"]}@{cfg["db_host"]}:{cfg["db_port"]}/{cfg["db_name"]}')
    # inspector = inspect(engine)
    # print ('Postgres database engine inspector created...')
    # schemas = inspector.get_schema_names()
    # print(schemas)
    # DBSession = sessionmaker(bind=engine)
    # with DBSession() as session:
    #     person = session.query(FacePerson).filter(FacePerson.name=='demo').one()
    # print(person)

    db = FaceDatabase()
    db.open(**cfg)
    db.register('阿兜', np.array([2.1, 2.2, 2.3]))
    f = db.features()
    # db.unregister_by_name('阿兜')
    # db.register('name2', [2.1, 2.2, 2.3])
    # db.unregister(2)
    # db.save('facedb.npz')
    # db.load('facedb.npz')

# arr1 = np.array([[1, 2, 3]])
# arr2 = np.array([[4, 5, 6]])
# arr = np.concatenate((arr1, arr2), axis=0)
# print(arr)