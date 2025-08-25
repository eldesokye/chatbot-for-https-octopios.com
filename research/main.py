import os
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import json
import requests
from datetime import datetime
import faiss
from dotenv import load_dotenv

# LangChain imports
from langchain.llms import GooglePalm
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough

# FastAPI imports
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Load environment variables
load_dotenv()

print("All libraries imported successfully!")


# ===== CELL 3: Configuration and Environment Setup =====
class ChatbotConfig:
    def __init__(self):
        # API Keys (you need to set these in your .env file)
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # Get from Google AI Studio
        self.WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")  # Get from OpenWeatherMap
        
        # File paths
        self.PRODUCTS_CSV_PATH = "products.csv"  # Your product catalog
        self.CUSTOMER_DATA_CSV_PATH = "customer_data.csv"  # Customer interaction history
        self.FAISS_INDEX_PATH = "product_index"
        
        # Memory settings
        self.MEMORY_WINDOW_SIZE = 10  # Number of conversation turns to remember
        
        # Model settings
        self.MODEL_NAME = "gemini-1.5-flash"  # Updated model name
        self.TEMPERATURE = 0.7
        
    def validate_config(self):
        """Validate that all required configurations are set"""
        if not self.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        print("Configuration validated successfully!")

config = ChatbotConfig()
config.validate_config()


# ===== CELL 5: Data Loader and Preprocessor =====
class DataManager:
    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.products_df = None
        self.customer_df = None
        self.load_data()
    
    def load_data(self):
        """Load CSV data files"""
        try:
            self.products_df = pd.read_csv(self.config.PRODUCTS_CSV_PATH)
            self.customer_df = pd.read_csv(self.config.CUSTOMER_DATA_CSV_PATH)
            print("Data loaded successfully!")
            print(f"Products: {len(self.products_df)} items")
            print(f"Customers: {len(self.customer_df)} records")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def get_customer_data(self, customer_id: str) -> Optional[Dict]:
        """Retrieve customer data by ID"""
        customer = self.customer_df[self.customer_df['customer_id'] == customer_id]
        if not customer.empty:
            return customer.iloc[0].to_dict()
        return None
    
    def get_products_by_criteria(self, weather: str = None, style: str = None, 
                               max_price: float = None) -> pd.DataFrame:
        """Filter products based on criteria"""
        filtered_products = self.products_df.copy()
        
        if weather:
            filtered_products = filtered_products[
                (filtered_products['weather_suitable'] == weather) | 
                (filtered_products['weather_suitable'] == 'any')
            ]
        
        if style:
            filtered_products = filtered_products[
                filtered_products['style'] == style
            ]
        
        if max_price:
            filtered_products = filtered_products[
                filtered_products['price'] <= max_price
            ]
        
        return filtered_products
    
    def create_product_documents(self) -> List[Document]:
        """Convert product data to LangChain documents for vector storage"""
        documents = []
        for _, product in self.products_df.iterrows():
            content = f"""
            Product: {product['name']}
            Category: {product['category']}
            Price: ${product['price']}
            Description: {product['description']}
            Weather Suitable: {product['weather_suitable']}
            Style: {product['style']}
            Stock: {product['stock']}
            Product ID: {product['product_id']}
            """
            doc = Document(
                page_content=content,
                metadata={
                    'product_id': product['product_id'],
                    'name': product['name'],
                    'price': product['price'],
                    'category': product['category']
                }
            )
            documents.append(doc)
        return documents

# Initialize data manager
data_manager = DataManager(config)
print("Data manager initialized!")


# ===== CELL 6: FAISS Vector Store Setup =====
class VectorStoreManager:
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.vector_store = None
        self.setup_vector_store()
    
    def setup_vector_store(self):
        """Create and populate FAISS vector store with product data"""
        print("Setting up FAISS vector store...")
        
        # Get product documents
        documents = self.data_manager.create_product_documents()
        
        # Split documents if they're too long
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        splits = text_splitter.split_documents(documents)
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        
        # Save the index
        self.vector_store.save_local(config.FAISS_INDEX_PATH)
        print("Vector store created and saved successfully!")
    
    def load_vector_store(self):
        """Load existing FAISS vector store"""
        try:
            self.vector_store = FAISS.load_local(
                config.FAISS_INDEX_PATH, 
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            print("Vector store loaded successfully!")
        except Exception as e:
            print(f"Error loading vector store: {e}")
            self.setup_vector_store()
    
    def search_products(self, query: str, k: int = 5) -> List[Document]:
        """Search for products using similarity search"""
        if self.vector_store:
            return self.vector_store.similarity_search(query, k=k)
        return []

# Initialize vector store
vector_manager = VectorStoreManager(data_manager)
print("Vector store manager initialized!")



# ===== CELL 7: External Tools (Weather API, Web Search) =====
class ExternalTools:
    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.weather_api_key = config.WEATHER_API_KEY
    
    def get_weather(self, location: str) -> Dict[str, Any]:
        """Get current weather for a location"""
        try:
            if not self.weather_api_key:
                return {"error": "Weather API key not configured"}
            
            url = f"http://api.openweathermap.org/data/2.5/weather"
            params = {
                'q': location,
                'appid': self.weather_api_key,
                'units': 'metric'
            }
            
            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                return {
                    'temperature': data['main']['temp'],
                    'condition': data['weather'][0]['main'].lower(),
                    'description': data['weather'][0]['description'],
                    'humidity': data['main']['humidity']
                }
            else:
                return {"error": "Failed to fetch weather data"}
        except Exception as e:
            return {"error": f"Weather API error: {str(e)}"}
    
    def web_search(self, query: str) -> str:
        """Simulate web search (you can integrate real search APIs here)"""
        # This is a placeholder - you can integrate with Google Search API, Bing, etc.
        return f"Search results for '{query}': Latest trends show increased demand for sustainable and eco-friendly products."

# Initialize external tools
external_tools = ExternalTools(config)
print("External tools initialized!")

# ===== CELL 8: Memory and Conversation Management =====
class ConversationManager:
    def __init__(self, config: ChatbotConfig):
        self.config = config
        self.conversations = {}  # Store conversations by customer ID
        
    def get_or_create_memory(self, customer_id: str) -> ConversationBufferWindowMemory:
        """Get existing memory or create new one for customer"""
        if customer_id not in self.conversations:
            self.conversations[customer_id] = ConversationBufferWindowMemory(
                k=self.config.MEMORY_WINDOW_SIZE,
                return_messages=True,
                memory_key="chat_history"
            )
        return self.conversations[customer_id]
    
    def save_conversation(self, customer_id: str, human_message: str, ai_message: str):
        """Save conversation to memory and optionally to CSV"""
        memory = self.get_or_create_memory(customer_id)
        
        # Add to conversation history CSV for long-term storage
        conversation_data = {
            'timestamp': [datetime.now().isoformat()],
            'customer_id': [customer_id],
            'human_message': [human_message],
            'ai_message': [ai_message]
        }
        
        conversation_df = pd.DataFrame(conversation_data)
        
        # Append to existing file or create new one
        try:
            existing_df = pd.read_csv("conversation_history.csv")
            updated_df = pd.concat([existing_df, conversation_df], ignore_index=True)
        except FileNotFoundError:
            updated_df = conversation_df
        
        updated_df.to_csv("conversation_history.csv", index=False)

# Initialize conversation manager
conversation_manager = ConversationManager(config)
print("Conversation manager initialized!")


# ===== CELL 9: Main Chatbot Logic with Gemini LLM =====
class EcommerceChatbot:
    def __init__(self, config: ChatbotConfig, data_manager: DataManager, 
                 vector_manager: VectorStoreManager, external_tools: ExternalTools,
                 conversation_manager: ConversationManager):
        self.config = config
        self.data_manager = data_manager
        self.vector_manager = vector_manager
        self.external_tools = external_tools
        self.conversation_manager = conversation_manager
        
        # Initialize Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            model=self.config.MODEL_NAME,
            temperature=self.config.TEMPERATURE,
            google_api_key=self.config.GOOGLE_API_KEY
        )
        
        # Create the main prompt template
        self.prompt_template = self._create_prompt_template()
        
    def _create_prompt_template(self) -> PromptTemplate:
        """Create the main prompt template for the chatbot"""
        template = """
        You are an intelligent e-commerce chatbot assistant designed to help customers find and purchase products.
        Your primary goal is to recommend products that match the customer's needs and encourage them to make a purchase.
        
        IMPORTANT GUIDELINES:
        - Be friendly, helpful, and enthusiastic about helping the customer
        - Use the customer's current answers and preferences to make personalized recommendations
        - Always try to guide the conversation towards making a purchase
        - Ask relevant questions to better understand customer needs
        - Use weather data and external context to make smart recommendations
        - Be persuasive but not pushy - focus on how products solve customer problems
        
        CUSTOMER CONTEXT:
        Customer ID: {customer_id}
        Customer Data: {customer_data}
        Current Weather: {weather_data}
        
        AVAILABLE PRODUCTS:
        {product_context}
        
        CONVERSATION HISTORY:
        {chat_history}
        
        CUSTOMER MESSAGE: {human_input}
        
        Respond as a helpful sales assistant. Make specific product recommendations with prices and explain why they're perfect for the customer.
        , and make the response maximun 3 lines for response .. 
        """
        
        return PromptTemplate(
            input_variables=["customer_id", "customer_data", "weather_data", 
                           "product_context", "chat_history", "human_input"],
            template=template
        )
    
    def get_product_recommendations(self, query: str, customer_data: Dict = None) -> str:
        """Get product recommendations based on query and customer data"""
        # Search for relevant products
        relevant_docs = self.vector_manager.search_products(query, k=5)
        
        # Format product information
        product_context = "\n".join([doc.page_content for doc in relevant_docs])
        
        return product_context
    
    def process_message(self, customer_id: str, message: str, location: str = None) -> str:
        """Process customer message and generate response"""
        try:
            # Get customer data
            customer_data = self.data_manager.get_customer_data(customer_id) or {}
            
            # Get weather data if location is provided
            weather_data = {}
            if location:
                weather_data = self.external_tools.get_weather(location)
            
            # Get conversation memory
            memory = self.conversation_manager.get_or_create_memory(customer_id)
            
            # Get product recommendations
            product_context = self.get_product_recommendations(message, customer_data)
            
            # Format chat history
            chat_history = ""
            if hasattr(memory, 'chat_memory') and memory.chat_memory.messages:
                for msg in memory.chat_memory.messages[-6:]:  # Last 3 exchanges
                    if hasattr(msg, 'content'):
                        role = "Human" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
                        chat_history += f"{role}: {msg.content}\n"
            
            # Create the prompt
            prompt_input = {
                "customer_id": customer_id,
                "customer_data": json.dumps(customer_data, default=str),
                "weather_data": json.dumps(weather_data, default=str),
                "product_context": product_context,
                "chat_history": chat_history,
                "human_input": message
            }
            
            # Generate response using Gemini
            formatted_prompt = self.prompt_template.format(**prompt_input)
            response = self.llm.invoke(formatted_prompt)
            
            # Extract the content from the response
            ai_message = response.content if hasattr(response, 'content') else str(response)
            
            # Save conversation
            memory.chat_memory.add_user_message(message)
            memory.chat_memory.add_ai_message(ai_message)
            self.conversation_manager.save_conversation(customer_id, message, ai_message)
            
            return ai_message
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}. How can I help you find the perfect products today?"
            return error_msg

# Initialize the main chatbot
chatbot = EcommerceChatbot(
    config=config,
    data_manager=data_manager,
    vector_manager=vector_manager,
    external_tools=external_tools,
    conversation_manager=conversation_manager
)
print("Main chatbot initialized!")



# ===== CELL 10: FastAPI Web Interface =====
# Pydantic models for API requests
class ChatRequest(BaseModel):
    customer_id: str
    message: str
    location: Optional[str] = None

class ChatResponse(BaseModel):
    customer_id: str
    response: str
    timestamp: str

# Initialize FastAPI app
app = FastAPI(
    title="E-commerce Chatbot API",
    description="AI-powered chatbot for product recommendations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "E-commerce Chatbot API is running!"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint"""
    try:
        # Process the message using our chatbot
        response = chatbot.process_message(
            customer_id=request.customer_id,
            message=request.message,
            location=request.location
        )
        
        return ChatResponse(
            customer_id=request.customer_id,
            response=response,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/customer/{customer_id}")
async def get_customer_data(customer_id: str):
    """Get customer data"""
    customer_data = data_manager.get_customer_data(customer_id)
    if customer_data:
        return customer_data
    else:
        raise HTTPException(status_code=404, detail="Customer not found")

@app.get("/products")
async def get_products(
    weather: Optional[str] = None,
    style: Optional[str] = None,
    max_price: Optional[float] = None
):
    """Get products with optional filtering"""
    products = data_manager.get_products_by_criteria(weather, style, max_price)
    return products.to_dict('records')

@app.post("/products/search")
async def search_products(query: str):
    """Search products using vector similarity"""
    results = vector_manager.search_products(query)
    return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]

print("FastAPI app configured!")


