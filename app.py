from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg_pool import AsyncConnectionPool
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker
from workflows.human_workflow import HumanWorkflow
from httpresponse.thread_response import ThreadResponse
from httpresponse.start_thread_response import StartThreadResponse
from sqlalchemy import Boolean, Column, String, Text
from httprequest.chat_request import ChatRequest
from httprequest.update_state_request import UpdateStateRequest


DEFAULT_DATABASE_URL = (
    "postgresql+psycopg://postgres:postgres@postgres_local:5432/postgres"
)
TARGET_DATABASE_URL = (
    "postgresql+psycopg://postgres:postgres@postgres_local:5432/threads_db"
)

default_engine = create_engine(DEFAULT_DATABASE_URL, future=True)
target_engine = create_engine(TARGET_DATABASE_URL, future=True)

Base = declarative_base()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=target_engine)

human_workflow = HumanWorkflow()

# Threads table
class Thread(Base):
    __tablename__ = "threads"
    thread_id = Column(String, primary_key=True, index=True)
    question_asked = Column(Boolean, default=False)
    question = Column(String, nullable=True)
    answer = Column(Text, nullable=True)
    confirmed = Column(Boolean, default=False)
    error = Column(Boolean, default=False)

# Create and initialise database
def initialize_database():
    with default_engine.connect() as connection:
        with connection.execution_options(isolation_level="AUTOCOMMIT"):
            result = connection.execute(
                text("SELECT 1 FROM pg_database WHERE datname = 'threads_db'")
            ).fetchone()
            if not result:
                connection.execute(text("CREATE DATABASE threads_db"))


def ensure_tables():
    Base.metadata.create_all(bind=target_engine)


# Method to get db connection, required for dependency injection
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    ensure_tables()
    conn_string = DEFAULT_DATABASE_URL.replace("postgresql+psycopg", "postgresql")

    async with AsyncConnectionPool(
        conninfo=conn_string,
        kwargs={"autocommit": True},
        max_size=20,
    ) as pool:
        checkpointer = AsyncPostgresSaver(pool)
        await checkpointer.setup()

        human_workflow.set_checkpointer(checkpointer)
        human_workflow.init_create_workflow()
        yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/start_thread", response_model=StartThreadResponse)
async def start_thread(db: Session = Depends(get_db)):
    """
    Create and start a thread

    Args:

    Returns:
        StartThreadResponse
    """
    thread_id = str(uuid4())
    new_thread = Thread(
        thread_id=thread_id, question_asked=False, confirmed=False, error=False
    )
    db.add(new_thread)
    db.commit()
    db.refresh(new_thread)
    return StartThreadResponse(thread_id=new_thread.thread_id)


@app.post("/article_writer/{thread_id}", response_model=ThreadResponse)
async def ask_question(
    thread_id: str, request: ChatRequest, db: Session = Depends(get_db)
):
    """
    Article writer

    Args:
        thread_id (str): Thread Id created in start thread API

    Returns:
        ThreadResponse: Object containing generated article 
    """
    thread = db.query(Thread).filter(Thread.thread_id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Thread ID does not exist.")
    if thread.question_asked:
        raise HTTPException(
            status_code=400,
            detail=f"Question has already been asked for thread ID: {thread_id}.",
        )
    if not request.sport_event:
        raise HTTPException(status_code=400, detail="Missing question.")
    response_state = await human_workflow.ainvoke(
        input={"event": request.sport_event},
        config={"recursion_limit": 15, "configurable": {"thread_id": thread_id}},
        subgraphs=True,
    )
    thread.question_asked = True
    thread.question = request.sport_event
    thread.answer = response_state[1].get("final_article")
    thread.error = response_state[1].get("error", False)
    db.commit()
    return ThreadResponse(
        thread_id=thread.thread_id,
        question_asked=thread.question_asked,
        question=thread.question,
        answer=thread.answer,
        confirmed=thread.confirmed,
        error=thread.error,
    )


@app.patch("/edit_state/{thread_id}", response_model=ThreadResponse)
async def edit_state(
    thread_id: str, request: UpdateStateRequest, db: Session = Depends(get_db)
):
    """
    Edit State for Human in the loop

    Args:
        thread_id (str): Thread ID associated with the article
        request (UpdateStateRequest): containing the edited article

    Response:
        ThreadResponse : Object containing generated article 
    """
    thread = db.query(Thread).filter(Thread.thread_id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Thread ID does not exist.")
    if not thread.question_asked:
        raise HTTPException(
            status_code=400, detail="Cannot edit a thread without a question."
        )
    if thread.confirmed:
        raise HTTPException(
            status_code=400, detail="Cannot edit a thread after it has been confirmed."
        )
    await human_workflow.workflow.aupdate_state(
        config={"configurable": {"thread_id": thread_id}},
        values={"answer": request.answer},
    )
    thread.answer = request.answer
    db.commit()
    return ThreadResponse(
        thread_id=thread.thread_id,
        question_asked=thread.question_asked,
        question=thread.question,
        answer=thread.answer,
        confirmed=thread.confirmed,
        error=thread.error,
    )


@app.post("/confirm/{thread_id}", response_model=ThreadResponse)
async def confirm(thread_id: str, db: Session = Depends(get_db)):
    """
    Confirm Article after human review

    Args:
        thread_id (str): thread id associated with the article
    
    Response:
        ThreadResponse: Object containing confirmed article
    """
    thread = db.query(Thread).filter(Thread.thread_id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Thread ID does not exist.")
    if not thread.question_asked:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot confirm thread {thread_id} as no question has been asked.",
        )
    response_state = await human_workflow.ainvoke(
        input=None,
        config={"configurable": {"thread_id": thread_id}},
    )
    thread.confirmed = bool(response_state.get("confirmed"))
    thread.answer = response_state.get("answer")
    db.commit()
    return ThreadResponse(
        thread_id=thread.thread_id,
        question_asked=thread.question_asked,
        question=thread.question,
        answer=thread.answer,
        confirmed=thread.confirmed,
        error=thread.error,
    )


@app.delete("/delete_thread/{thread_id}", response_model=ThreadResponse)
async def delete_thread(thread_id: str, db: Session = Depends(get_db)):
    """
    Deleting a thread

    Args:
        thread_id (str): thread id associated with the article

    Response:
        ThreadResponse: Object containing the deleted article
    """
    thread = db.query(Thread).filter(Thread.thread_id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Thread ID does not exist.")
    db.delete(thread)
    db.commit()
    return ThreadResponse(
        thread_id=thread.thread_id,
        question_asked=thread.question_asked,
        question=thread.question,
        answer=thread.answer,
        confirmed=thread.confirmed,
        error=thread.error,
    )


@app.get("/sessions", response_model=list[ThreadResponse])
async def list_sessions(db: Session = Depends(get_db)):
    """
    List all sessions or threads or articles

    Args:
        none
    
    Response:
        List[ThreadResponse] : List of all active articles or threads
    """
    threads = db.query(Thread).all()
    return [
        ThreadResponse(
            thread_id=thread.thread_id,
            question_asked=thread.question_asked,
            question=thread.question,
            answer=thread.answer,
            confirmed=thread.confirmed,
            error=thread.error,
        )
        for thread in threads
    ]
