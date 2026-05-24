"""
Neo4j Database Bağlantı Yöneticisi

Bu modül, Neo4j veritabanına bağlantıyı yönetir.
Singleton pattern kullanarak tek bir driver instance'ı oluşturur.
"""

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from contextlib import contextmanager
from typing import Optional, Generator

from tck_graphrag.core.config import get_settings


class Neo4jConnection:
    """
    Neo4j veritabanı bağlantı yöneticisi.
    
    Kullanım:
        neo4j = Neo4jConnection()
        neo4j.connect()
        
        # Sorgu çalıştır
        with neo4j.get_session() as session:
            result = session.run("MATCH (n) RETURN count(n)")
            
        neo4j.close()
    """
    
    _instance: Optional['Neo4jConnection'] = None
    
    def __new__(cls) -> 'Neo4jConnection':
        """Singleton pattern - tek instance"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self.settings = get_settings()
        self._driver = None
        self._initialized = True
    
    def connect(self) -> bool:
        """
        Neo4j'e bağlan ve bağlantıyı test et.
        
        Returns:
            bool: Bağlantı başarılı ise True
        """
        try:
            self._driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_username, self.settings.neo4j_password)
            )
            
            # Bağlantıyı test et
            self._driver.verify_connectivity()
            print(f"✅ Neo4j bağlantısı başarılı: {self.settings.neo4j_uri}")
            return True
            
        except AuthError:
            print(f"❌ Neo4j kimlik doğrulama hatası! Şifrenizi kontrol edin.")
            return False
        except ServiceUnavailable:
            print(f"❌ Neo4j servisine ulaşılamıyor: {self.settings.neo4j_uri}")
            print("   Neo4j Desktop'ta instance'ın 'Running' durumunda olduğundan emin olun.")
            return False
        except Exception as e:
            print(f"❌ Neo4j bağlantı hatası: {e}")
            return False
    
    def close(self):
        """Bağlantıyı kapat."""
        if self._driver:
            self._driver.close()
            self._driver = None
            print("Neo4j bağlantısı kapatıldı.")
    
    @property
    def driver(self):
        """Neo4j driver instance'ı döner."""
        return self._driver
    
    @contextmanager
    def get_session(self, database: str = "neo4j") -> Generator:
        """
        Context manager ile session al.
        
        Args:
            database: Kullanılacak veritabanı adı (varsayılan: neo4j)
            
        Yields:
            Neo4j Session objesi
        """
        if not self._driver:
            raise RuntimeError("Neo4j'e bağlı değil! Önce connect() çağırın.")
            
        session = self._driver.session(database=database)
        try:
            yield session
        finally:
            session.close()
    
    def run_query(self, query: str, parameters: dict = None) -> list:
        """
        Cypher sorgusu çalıştır ve sonuçları döndür.
        
        Args:
            query: Cypher sorgusu
            parameters: Sorgu parametreleri (opsiyonel)
            
        Returns:
            Sonuç listesi
        """
        with self.get_session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]
    
    def get_database_info(self) -> dict:
        """
        Veritabanı bilgilerini döner.
        
        Returns:
            Node ve relationship sayıları
        """
        node_count = self.run_query("MATCH (n) RETURN count(n) as count")[0]["count"]
        rel_count = self.run_query("MATCH ()-[r]->() RETURN count(r) as count")[0]["count"]
        
        return {
            "node_count": node_count,
            "relationship_count": rel_count
        }


# Kolay erişim için global instance
def get_neo4j() -> Neo4jConnection:
    """Neo4j bağlantı instance'ı döner."""
    return Neo4jConnection()
