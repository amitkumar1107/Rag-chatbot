import langchain
print("LangChain version:", langchain.__version__)
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity 
from langchain_ollama import ChatOllama
import json

app = FastAPI(title="Amit Profile Assistant API")

# Lets the Vite development server call this API from a different local port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)

llm = ChatOllama(
    model="qwen3:0.6b",
    temperature=0.3
)

model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
personal_profile = [
    {"amit's best friends": "Amit's best friend is Ankit"},
    {
        "category": "personality_observations",
        "source": "Observed from conversations",
        "traits": [
            {
                "trait": "Career-oriented",
                "description": (
                    "Frequently thinks about career growth, new opportunities, "
                    "and building a stronger professional future."
                )
            },
            {
                "trait": "Curious",
                "description": (
                    "Asks questions across backend engineering, artificial intelligence, "
                    "product development, business ideas, and professional growth."
                )
            },
            {
                "trait": "Practical thinker",
                "description": (
                    "Often focuses on how ideas can be implemented in real applications "
                    "and how technology can solve practical problems."
                )
            },
            {
                "trait": "Self-improvement oriented",
                "description": (
                    "Regularly looks for ways to improve technical knowledge, "
                    "communication, interview skills, and professional presentation."
                )
            },
            {
                "trait": "Entrepreneurially curious",
                "description": (
                    "Explores software business ideas and considers how applications "
                    "could become useful products or businesses."
                )
            },
            {
                "trait": "Persistent problem-solver",
                "description": (
                    "Often follows up on questions and looks for alternative approaches "
                    "when an initial solution does not meet expectations."
                )
            }
        ]
    },
    {
        "category": "learning_preferences",
        "source": "Directly demonstrated in conversations",
        "preferences": [
            "Prefers simple and clear English.",
            "Likes step-by-step explanations.",
            "Values practical examples over theory alone.",
            "Likes structured learning roadmaps.",
            "Benefits from hands-on projects related to real-world applications.",
            "Often wants to understand how technologies work together in a complete system.",
            "Is interested in interview-style questions and practical preparation."
        ]
    },
    {
        "category": "career_mindset",
        "source": "Directly shared goals and recurring topics",
        "details": [
            "Wants to build on existing Python backend experience.",
            "Is interested in transitioning toward AI engineering.",
            "Wants to develop stronger technical interview skills.",
            "Is interested in learning modern AI development techniques.",
            "Values practical skills that can be demonstrated through projects.",
            "Is interested in improving career opportunities through continued learning."
        ]
    },
    {
        "category": "technology_interests",
        "source": "Directly discussed interests",
        "topics": [
            "Python backend engineering",
            "Django and REST API development",
            "Generative AI",
            "Large language model applications",
            "Retrieval-Augmented Generation",
            "AI-powered automation",
            "Real-time AI applications",
            "Speech-enabled AI systems",
            "Full-stack product development",
            "Software architecture and system design"
        ]
    },
    {
        "category": "product_and_business_interests",
        "source": "Ideas discussed; not necessarily launched",
        "interests": [
            {
                "area": "AI-powered products",
                "description": (
                    "Interested in building applications that use AI to automate "
                    "workflows and provide useful features."
                )
            },
            {
                "area": "Influencer marketing",
                "description": (
                    "Explored a platform concept that connects influencers, brands, "
                    "and marketing agencies through campaign discovery and matching."
                )
            },
            {
                "area": "Website development business",
                "description": (
                    "Has explored the idea of offering website development services "
                    "and building a business around software solutions."
                )
            },
            {
                "area": "Product design",
                "description": (
                    "Cares about creating complete application experiences, including "
                    "landing pages, onboarding, role-based dashboards, and polished UI."
                )
            }
        ],
        "status_note": (
            "These are interests and exploratory ideas, not confirmation that a business "
            "or product has launched."
        )
    },
    {
        "category": "communication_style",
        "source": "Observed from conversations",
        "details": [
            "Prefers friendly, informal conversation.",
            "Usually prefers direct answers rather than unnecessarily complicated language.",
            "Often asks for ready-to-use examples, templates, and implementation guidance.",
            "Is interested in communicating more confidently in professional situations."
        ]
    },
    {
        "category": "personal_interests",
        "source": "Topics voluntarily discussed",
        "interests": [
            {
                "topic": "Fitness",
                "description": (
                    "Has discussed gym routines, exercise, and fitness-related goals."
                )
            },
            {
                "topic": "Professional development",
                "description": (
                    "Regularly explores ways to improve technical and workplace skills."
                )
            },
            {
                "topic": "Technology and innovation",
                "description": (
                    "Shows interest in new AI tools, software products, and ways "
                    "to use technology to solve problems."
                )
            }
        ]
    },
    {
        "category": "working_preferences",
        "source": "Observed from requests",
        "preferences": [
            "Likes actionable next steps.",
            "Prefers solutions that can be implemented rather than only discussed.",
            "Often wants complete workflows instead of isolated features.",
            "Values polished, professional results when building products.",
            "Appreciates explanations that connect technical concepts to business use cases."
        ]
    },
    {
        "category": "profile_accuracy_rules",
        "rules": [
            "Personality descriptions are conversational observations, not verified facts.",
            "Do not infer psychological conditions or make personality diagnoses.",
            "Do not turn a question about a topic into a claim that the user is an expert in it.",
            "Do not describe an idea as a launched business or product without confirmation.",
            "Do not publish private struggles or sensitive personal information.",
            "Do not include education details.",
            "When uncertain, use phrases such as 'Amit has expressed interest in' rather than 'Amit is an expert in'."
        ]
    },
    {
        "category": "basic_profile",
        "name": "Amit Kumar",
        "professional_title": "Python Backend Developer",
        "country": "India",
        "experience": "Approximately 2 years of Python backend development experience as of late 2026"
    },
    {
        "category": "professional_summary",
        "details": [
            "Python Backend Developer experienced in building backend APIs and systems.",
            "Experienced with Python, Django, Django REST Framework, MongoDB, and PostgreSQL.",
            "Has worked on API development, performance optimization, caching, background jobs, and real-time communication.",
            "Has professional experience contributing to AI-enabled recruitment and interview products.",
            "Interested in growing into an AI Engineer or Generative AI Engineer role."
        ]
    },
    {
        "category": "employment",
        "company": "Marktine Technology Solutions Pvt Ltd",
        "role": "Python Developer / Junior Software Engineer",
        "start_date": "2024-12-02",
        "responsibilities": [
            "Developing backend APIs using Python and Django technologies.",
            "Working with Django REST Framework.",
            "Working with MongoDB and PostgreSQL.",
            "Implementing API performance improvements and caching.",
            "Working with Celery and Redis for background processing.",
            "Contributing to AI-enabled recruitment and interview workflows.",
            "Integrating real-time communication and speech services."
        ]
    },
    {
        "category": "technical_skills",
        "programming_languages": [
            "Python"
        ],
        "backend_frameworks": [
            "Django",
            "Django REST Framework"
        ],
        "api_and_communication": [
            "REST APIs",
            "Django WebSockets",
            "Real-time communication"
        ],
        "databases": [
            "MongoDB",
            "PostgreSQL"
        ],
        "caching_and_background_jobs": [
            "Redis",
            "Celery",
            "Background schedulers",
            "Caching"
        ],
        "tools": [
            "Git",
            "Docker"
        ],
        "ai_and_speech_integrations": [
            "OpenAI GPT-4o",
            "Azure Speech-to-Text",
            "Azure Text-to-Speech"
        ],
        "learning_and_exploration": [
            "Retrieval-Augmented Generation (RAG)",
            "LangChain",
            "Generative AI",
            "LLM applications",
            "AI agents"
        ]
    },
    {
        "category": "project",
        "name": "AI-enabled recruitment platform",
        "type": "Professional project experience",
        "description": (
            "A recruitment platform supporting recruiter and candidate workflows, "
            "job management, candidate management, communication, and recruitment operations."
        ),
        "features_and_work_areas": [
            "Backend API development",
            "Job post approval workflows",
            "Candidate dashboard APIs",
            "Job description cloning",
            "Job assignment and versioning",
            "Caching and performance optimization",
            "Recruitment communication and automation"
        ],
        "public_disclosure": (
            "Confirm employer approval before publishing project names, client details, "
            "internal architecture, or confidential implementation details."
        )
    },
    {
        "category": "project",
        "name": "AI interview platform",
        "type": "Professional project experience",
        "description": (
            "An AI-assisted interview platform using language-model and speech services "
            "to support real-time interview interactions."
        ),
        "technologies": [
            "Python",
            "Django",
            "OpenAI GPT-4o",
            "Azure Speech-to-Text",
            "Azure Text-to-Speech",
            "Django WebSockets"
        ],
        "work_areas": [
            "AI interview backend integration",
            "Real-time communication",
            "Speech input and output integration",
            "Speech chunk handling",
            "Turn-taking behavior"
        ],
        "public_disclosure": (
            "Do not publish confidential prompts, code, internal architecture, "
            "or product screenshots without authorization."
        )
    },
    {
        "category": "career_goals",
        "primary_goal": "Become an AI Engineer or Generative AI Engineer",
        "background_to_build_on": "Python backend development",
        "interests": [
            "Generative AI",
            "RAG systems",
            "LLM application development",
            "AI agents and workflows",
            "System design",
            "Backend engineering",
            "Technical interview preparation",
            "Problem solving"
        ],
        "preferred_learning_style": [
            "Practical learning",
            "Hands-on projects",
            "Simple explanations",
            "Real-world applications"
        ]
    },
    {
        "category": "personal_product_idea",
        "name": "Influencer-brand collaboration platform",
        "status": "Concept under exploration; not confirmed as launched",
        "description": (
            "A platform concept connecting influencers with brands and marketing agencies "
            "for campaign discovery, applications, and collaboration management."
        ),
        "potential_features": [
            "Influencer profiles",
            "Company and agency profiles",
            "Social account analytics",
            "Brand campaign listings",
            "Influencer applications",
            "Brand-to-influencer matching scores",
            "Campaign invitations",
            "Application status tracking",
            "AI-assisted influencer matching",
            "Content planning and post scheduling"
        ]
    },
    {
        "category": "communication_preferences",
        "language": "English",
        "preferred_style": [
            "Simple",
            "Clear",
            "Practical",
            "Easy-to-understand explanations"
        ],
        "professional_topics": [
            "Python backend development",
            "Django and REST APIs",
            "AI and LLM applications",
            "RAG chatbot development",
            "Career development",
            "Technical interviews"
        ]
    },
    {
        "category": "chatbot_privacy_rules",
        "rules": [
            "Answer only using approved knowledge-base information.",
            "Do not invent personal details or professional achievements.",
            "Do not disclose education details.",
            "Do not reveal private financial, family, or health information.",
            "Distinguish professional experience from learning interests and future goals.",
            "Do not disclose confidential employer or client information.",
            "Do not claim that an experimental product has launched.",
            "If information is unavailable, say it is not included in the public profile."
        ]
    }
]
profile_texts = [
    json.dumps(profile, ensure_ascii=False)
    for profile in personal_profile
]

# Generate embeddings
embeddings = model.embed_documents(profile_texts)
print("Number of embeddings:", len(embeddings))

# app.get("/user-query")
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

def user_query(query: str):

    # Step 1: Convert user query into an embedding
    query_embedding = model.embed_query(query)

    # Step 2: Calculate similarity with stored embeddings
    similarities = cosine_similarity(
        [query_embedding],
        embeddings
    )[0]

    # Step 3: Get top 3 most relevant documents
    top_indices = np.argsort(similarities)[::-1][:3]

    # Step 4: Retrieve relevant profile documents
    relevant_context = []

    for index in top_indices:
        relevant_context.append(profile_texts[index])

    context = "\n".join(relevant_context)

    # Step 5: Create prompt
    prompt = f"""
    You are Amit AI, a friendly, helpful conversational assistant.

    Have natural conversations: greet the user warmly, answer casual questions,
    and help with general topics such as programming, learning, career advice,
    technology, and everyday questions in clear, simple English.

    The context below is Amit's approved public profile. Use it only when the
    user asks about Amit, his experience, skills, projects, goals, or other
    personal details. Never invent personal facts; if that personal information
    is not in the context, say that it is not included in the public profile.

    For general questions that are not about Amit, answer normally from your
    general knowledge. Do not claim that general information is about Amit.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    # Step 6: Generate response using Ollama
    llm_response = llm.invoke(prompt)

    return {
        "query": query,
        "similarities": similarities.tolist(),
        "retrieved_documents": relevant_context,
        "llm_response": llm_response.content
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Answer a frontend chat query using the local RAG profile."""
    try:
        result = await run_in_threadpool(user_query, request.query.strip())
        return {
            "answer": result["llm_response"],
            "sources": result["retrieved_documents"],
        }
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="The local AI service is unavailable. Ensure Ollama is running and the qwen3:0.6b model is installed.",
        ) from exc
