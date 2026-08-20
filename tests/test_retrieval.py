import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add project root directory to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval.models import RetrievalResult
from app.retrieval.services import (
    retrieve_text,
    retrieve_images,
    retrieve_multimodal,
    map_text_hit,
    map_image_hit
)
from app.retrieval.langgraph_integration import retrieval_node, RetrievalState
from app.qdrant.client import get_qdrant_client
from app.qdrant.collections import (
    create_omnibrain_collections,
    TEXT_COLLECTION,
    IMAGE_COLLECTION
)
from app.qdrant.insert import insert_text_vector, insert_image_vector
from PIL import Image

# Helper mock classes for Qdrant responses
class MockPoint:
    def __init__(self, id, score, payload):
        self.id = id
        self.score = score
        self.payload = payload

class MockResponse:
    def __init__(self, points):
        self.points = points


class TestRetrievalUnit(unittest.TestCase):
    """
    Unit tests for the retrieval layer using mocks to ensure offline isolation
    and correctness across edge cases, validation, and LangGraph state safety.
    """

    def setUp(self):
        self.mock_client = MagicMock()
        
    @patch('app.retrieval.services.search_text_similarity')
    def test_test1_text_query_retrieval(self, mock_search):
        """TEST 1: Text query -> text retrieval -> Qdrant -> structured results"""
        mock_search.return_value = [
            {
                "id": 1,
                "score": 0.95,
                "payload": {
                    "document_name": "annual_report.pdf",
                    "page_number": 25,
                    "chunk_id": "chunk_025_01",
                    "source_path": "data/documents/annual_report.pdf",
                    "text": "Revenue increased by 25 percent in 2025."
                }
            }
        ]
        
        results = retrieve_text("revenue growth", top_k=1, client=self.mock_client)
        
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], RetrievalResult)
        self.assertEqual(results[0].modality, "text")
        self.assertEqual(results[0].score, 0.95)
        self.assertEqual(results[0].document_name, "annual_report.pdf")
        self.assertEqual(results[0].page_number, 25)
        self.assertEqual(results[0].chunk_id, "chunk_025_01")
        self.assertEqual(results[0].content, "Revenue increased by 25 percent in 2025.")
        
    @patch('app.retrieval.services.get_image_model')
    def test_test2_image_query_retrieval(self, mock_get_model):
        """TEST 2: Image-related query -> image retrieval -> Qdrant -> structured results"""
        # Mock CLIP model encoding
        mock_model = MagicMock()
        mock_model.encode.return_value = [0.1] * 512
        mock_get_model.return_value = mock_model
        
        # Mock Qdrant response points
        mock_points = [
            MockPoint(101, 0.85, {
                "document_name": "annual_report.pdf",
                "page_number": 25,
                "image_id": "image_025_01",
                "source_path": "data/images/sample.png"
            })
        ]
        self.mock_client.query_points.return_value = MockResponse(mock_points)
        
        # Perform retrieval using text query on image collection
        results = retrieve_images("green chart image", top_k=1, client=self.mock_client)
        
        self.assertEqual(len(results), 1)
        self.assertIsInstance(results[0], RetrievalResult)
        self.assertEqual(results[0].modality, "image")
        self.assertEqual(results[0].score, 0.85)
        self.assertEqual(results[0].image_id, "image_025_01")
        self.assertEqual(results[0].image_path, "data/images/sample.png")
        
    @patch('app.retrieval.services.retrieve_images')
    @patch('app.retrieval.services.retrieve_text')
    def test_test3_multimodal_retrieval(self, mock_retrieve_text, mock_retrieve_images):
        """TEST 3: Multimodal retrieval -> text + image results"""
        mock_retrieve_text.return_value = [
            RetrievalResult(
                id=1, modality="text", score=0.90, document_name="doc.pdf",
                page_number=1, chunk_id="c1", source_path="doc.pdf", content="text"
            )
        ]
        mock_retrieve_images.return_value = [
            RetrievalResult(
                id=101, modality="image", score=0.95, document_name="doc.pdf",
                page_number=1, image_id="i1", source_path="img.png", image_path="img.png"
            )
        ]
        
        results = retrieve_multimodal("query", top_k=2, client=self.mock_client)
        
        self.assertEqual(len(results), 2)
        # Verify ordering by score descending: image score 0.95 first, then text score 0.90
        self.assertEqual(results[0].modality, "image")
        self.assertEqual(results[1].modality, "text")
        
    @patch('app.retrieval.services.search_text_similarity')
    def test_test4_query_no_relevant_results(self, mock_search):
        """TEST 4: Query with no relevant results"""
        mock_search.return_value = []
        results = retrieve_text("some obscure topic", top_k=3, client=self.mock_client)
        self.assertEqual(results, [])
        
    def test_test5_empty_query(self):
        """TEST 5: Empty query verification"""
        with self.assertRaises(ValueError):
            retrieve_text("", client=self.mock_client)
        with self.assertRaises(ValueError):
            retrieve_images("   ", client=self.mock_client)
            
    def test_test6_invalid_missing_metadata(self):
        """TEST 6: Invalid/missing metadata handling"""
        # If payload is empty or has missing keys, mapping should not fail
        payload = {}
        res_text = map_text_hit(1, 0.8, payload)
        self.assertEqual(res_text.document_name, "Unknown")
        self.assertEqual(res_text.page_number, -1)
        self.assertEqual(res_text.content, "")
        
        res_img = map_image_hit(101, 0.7, None)
        self.assertEqual(res_img.document_name, "Unknown")
        self.assertEqual(res_img.image_id, None)
        
    def test_test7_qdrant_unavailable_handling(self):
        """TEST 7: Qdrant unavailable/error handling"""
        self.mock_client.query_points.side_effect = Exception("Qdrant connection refused")
        with self.assertRaises(RuntimeError):
            # Because retrieve_images calls model.encode first, let's bypass it by testing retrieve_images with an existing path or stub
            retrieve_images("data/images/sample.png", client=self.mock_client)
            
    @patch('app.retrieval.langgraph_integration.retrieve_text')
    def test_test8_langgraph_state_receives_results(self, mock_retrieve_text):
        """TEST 8: LangGraph state receives retrieval results correctly"""
        mock_retrieve_text.return_value = [
            RetrievalResult(
                id=1, modality="text", score=0.90, document_name="doc.pdf",
                page_number=1, chunk_id="c1", source_path="doc.pdf", content="text content"
            )
        ]
        
        # Define mock state with user_query and retrieval_mode inputs
        state: RetrievalState = {
            "user_query": "revenue",
            "retrieval_mode": "text",
            "top_k": 3,
            "conversation_state": {"user_id": 123}
        }
        
        # Execute node
        updates = retrieval_node(state)
        
        # Verify updates are returned
        self.assertIn("retrieved_text", updates)
        self.assertIn("retrieved_images", updates)
        self.assertIn("retrieval_results", updates)
        self.assertIn("retrieved_context", updates)
        self.assertIn("similarity_scores", updates)
        self.assertIn("document_name", updates)
        self.assertIn("page_number", updates)
        self.assertIn("chunk_id", updates)
        self.assertIn("source_path", updates)
        self.assertEqual(updates["retrieval_status"], "success")
        self.assertEqual(updates["error_message"], "")
        
        # Verify state safety: original fields in state are unaffected
        self.assertEqual(state["user_query"], "revenue")
        self.assertEqual(state["retrieval_mode"], "text")
        self.assertEqual(state["conversation_state"]["user_id"], 123)
        
        # Check results content
        results = updates["retrieval_results"]
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], 1)
        self.assertEqual(results[0]["type"], "text")
        self.assertEqual(results[0]["content"], "text content")
        self.assertIn("doc.pdf", updates["retrieved_context"])
        
        self.assertEqual(updates["document_name"], ["doc.pdf"])
        self.assertEqual(updates["page_number"], [1])
        self.assertEqual(updates["chunk_id"], ["c1"])
        
    @patch('app.retrieval.langgraph_integration.retrieve_text')
    def test_langgraph_node_error_handling(self, mock_retrieve_text):
        """Test that retrieval_node handles exceptions safely by returning an error status"""
        mock_retrieve_text.side_effect = Exception("Mocked Qdrant Failure")
        
        state: RetrievalState = {
            "user_query": "revenue",
            "retrieval_mode": "text"
        }
        
        updates = retrieval_node(state)
        
        self.assertEqual(updates["retrieval_status"], "error")
        self.assertEqual(updates["error_message"], "Mocked Qdrant Failure")
        self.assertEqual(updates["retrieved_text"], [])
        self.assertEqual(updates["retrieved_images"], [])

        
    @patch('app.retrieval.services.search_text_similarity')
    def test_test9_top_k_limit(self, mock_search):
        """TEST 9: Top-k limit works correctly"""
        mock_search.return_value = [
            {"id": 1, "score": 0.9, "payload": {}},
            {"id": 2, "score": 0.8, "payload": {}},
            {"id": 3, "score": 0.7, "payload": {}}
        ]
        
        results = retrieve_text("test", top_k=2, client=self.mock_client)
        # search_text_similarity is called with top_k=2, which returns up to 2 items
        mock_search.assert_called_with(self.mock_client, "test", top_k=2)
        
    @patch('app.retrieval.services.search_text_similarity')
    def test_test10_similarity_scores_returned(self, mock_search):
        """TEST 10: Similarity scores are returned correctly"""
        mock_search.return_value = [
            {"id": 1, "score": 0.8877, "payload": {}}
        ]
        results = retrieve_text("test", top_k=1, client=self.mock_client)
        self.assertEqual(results[0].score, 0.8877)


class TestRetrievalRealIntegration(unittest.TestCase):
    """
    Real integration tests executing search on actual local-disk Qdrant storage.
    Verifies CLIP embedding models and Qdrant persistence end-to-end.
    """

    def setUp(self):
        # Set up real disk-based client in an isolated test database folder
        import shutil
        self.test_db_dir = os.path.join("data", "test_qdrant_db")
        if os.path.exists(self.test_db_dir):
            try:
                shutil.rmtree(self.test_db_dir)
            except Exception:
                pass
        
        from qdrant_client import QdrantClient
        self.client = QdrantClient(path=self.test_db_dir)
        create_omnibrain_collections(self.client)
        
    def tearDown(self):
        # Close connection to release file locks and clean up test directory
        if hasattr(self, "client"):
            try:
                self.client.close()
            except Exception:
                pass
        import shutil
        if os.path.exists(self.test_db_dir):
            try:
                shutil.rmtree(self.test_db_dir)
            except Exception:
                pass

    def test_real_end_to_end_search(self):
        """Step 9: Real end-to-end text and image search validation using real embeddings & Qdrant"""
        print("\n--- Running Real End-To-End Integration Test ---")
        
        # 1. Insert Text Vector
        text = "The company's revenue increased by 25 percent in 2025 due to strong market growth."
        from app.embeddings.text_embeddings import generate_text_embedding
        text_vector = generate_text_embedding(text)
        
        insert_text_vector(
            client=self.client,
            point_id=1,
            vector=text_vector,
            document_name="annual_report.pdf",
            page_number=25,
            chunk_id="chunk_025_01",
            source_path="data/documents/annual_report.pdf",
            text=text
        )
        
        # 2. Retrieve Text and Verify
        results = retrieve_text("revenue growth", top_k=1, client=self.client)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].document_name, "annual_report.pdf")
        self.assertEqual(results[0].page_number, 25)
        self.assertEqual(results[0].chunk_id, "chunk_025_01")
        self.assertEqual(results[0].content, text)
        self.assertGreater(results[0].score, 0.0) # Cosine similarity should be positive for match
        
        # 3. Create, Ingest and Retrieve Image
        os.makedirs(os.path.join("data", "images"), exist_ok=True)
        img_path = os.path.join("data", "images", "integration_sample.png")
        # Solid red image
        img = Image.new("RGB", (224, 224), color=(220, 20, 60))
        img.save(img_path)
        
        from app.embeddings.image_embeddings import generate_image_embedding
        img_vector = generate_image_embedding(img_path)
        
        insert_image_vector(
            client=self.client,
            point_id=101,
            vector=img_vector,
            document_name="annual_report.pdf",
            page_number=25,
            image_id="image_025_01",
            source_path=img_path
        )
        
        # Test text-to-image search: Query image using description
        img_results = retrieve_images("a red square", top_k=1, client=self.client)
        self.assertEqual(len(img_results), 1)
        self.assertEqual(img_results[0].image_id, "image_025_01")
        self.assertEqual(img_results[0].image_path, img_path)
        
        # Cleanup image
        if os.path.exists(img_path):
            os.remove(img_path)



if __name__ == "__main__":
    unittest.main()
