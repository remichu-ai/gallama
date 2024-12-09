# using absolute import here as this file will be run alone
import torch
import gc
from fastapi import FastAPI, Request, APIRouter, File, UploadFile, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from openai.types.beta.assistant_stream_event import ThreadRunStepExpired
import io
import soundfile as sf
from gallama.data_classes import (
    ChatMLQuery,
    ToolForce,
    GenerateQuery,
    ModelObjectResponse,
    ModelObject,
    ModelParser,
    GenQueue,
    EmbeddingRequest,
    TranscriptionResponse,
    TTSRequest
)
import argparse
from gallama.backend.llm import ModelExllama, ModelLlamaCpp, ModelTransformers
from gallama.backend.llm.prompt_engine import PromptEngine
import uvicorn
from fastapi.exceptions import RequestValidationError
from sse_starlette.sse import EventSourceResponse
import json
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from gallama.api_response.chat_response import (
    chat_completion_response_stream,
    completion_response_stream,
    completion_response,
    chat_completion_response,
    chat_completion_response_artifact_stream,
    chat_completion_response_artifact
)
from gallama.utils import parse_request_body
from gallama.config.config_manager import ConfigManager
from gallama.data import ARTIFACT_SYSTEM_PROMPT
from typing import List, Literal, Optional
import os
import asyncio
from contextlib import asynccontextmanager
from RealtimeTTS import ParlerEngine
import threading
from asyncio import Queue
from logging import DEBUG
from gallama.logger.logger import get_logger
# Add this after your imports to clear logging from 3rd party module

#streaming example
#https://blog.gopenai.com/how-to-stream-output-in-llm-based-applications-to-boost-user-experience-e9fcf582777a

logger = get_logger()
router = APIRouter()
config_manager = ConfigManager()
model_ready = False


@router.get("/")
async def read_root():
    return {"Hello": "World"}


def validate_api_request(query: ChatMLQuery):
    # fixing api call parameter
    # update temperature to 0.1 because huggingface doesn't accept 0 temperature
    if query.temperature <= 0:
        query.temperature = 0.01
    # tool_choice is set to 'auto' if there is tool

    # set tool_choice if user didn't set it and ensure parameters are correct
    # if no tool, set tool_choice to none
    if query.tools:
        if query.tool_choice is None:
            if query.tools:
                query.tool_choice = "auto"
            else:
                query.tool_choice = "none"
        elif query.tool_choice == "none":
            # wipe out all the tools if any
            query.tools = None
        elif isinstance(query.tool_choice, ToolForce):
            # wipe out all the tool except the tool to be forced 'tool_choice' in api call
            force_single_tool = []
            for tool in query.tools:
                if tool.function.name == query.tool_choice.function.name:
                    force_single_tool.append(tool)

            if len(force_single_tool) != 1:
                raise Exception("tool_choice not exist in function list")
            else:
                query.tool_choice = "required"
                query.tools = force_single_tool
    else:
        query.tool_choice = "none"

    # validate that prefix_strings and regex_prefix_pattern  or regex_pattern can not be used together
    if query.prefix_strings and (query.regex_pattern or query.regex_prefix_pattern):
        raise HTTPException(status_code=400, detail="refix_strings and regex_pattern, regex_prefix_pattern can not be used together")

    return query


async def result_generator(gen_queue):
    completed_event = asyncio.Event()

    while True:
        result = await gen_queue.get()
        if result == {"data": "[DONE]"}:
            completed_event.set()
            yield result
            break

        yield {"data": result.json()}

    # Wait until the task is marked as complete
    await completed_event.wait()


# noinspection PyAsyncCall
@router.post("/v1/chat/completions")
async def chat_completion(request: Request, query: ChatMLQuery):
    # https://platform.openai.com/docs/api-reference/chat/create

    global llm_dict, default_model_name
    gen_queue = GenQueue()      # this queue will hold the result for this generation

    try:
        # validate and fix query
        query = validate_api_request(query)

        if llm_dict.get(query.model):
            model_name_to_use = query.model
        else:
            model_name_to_use = default_model_name

        llm = llm_dict[model_name_to_use]["model"]
        prompt_eng = llm_dict[model_name_to_use]["prompt_engine"]

        # log if thinking is used
        if query.thinking_template:
            logger.info(f"thinking is used with returnThinking set to {query.return_thinking}")

        # start the generation task
        asyncio.create_task(
            llm.chat(
            query=query,
            prompt_eng=prompt_eng,
            gen_queue=gen_queue,
            request=request,
        ))

        # send the response to client
        if query.stream:
            # EventSourceResponse take iterator so need to handle at here
            if query.artifact == "No":     # not using artefact
                return EventSourceResponse(
                    chat_completion_response_stream(
                        query=query, gen_queue=gen_queue, model_name=model_name_to_use, request=request,
                    ))
            else:
                return EventSourceResponse(
                    chat_completion_response_artifact_stream(
                        query=query, gen_queue=gen_queue, model_name=model_name_to_use, request=request,
                    ))
        else:
            if query.artifact == "No":     # not using artefact
                return await chat_completion_response(query=query, gen_queue=gen_queue, model_name=model_name_to_use, request=request,)
            else:
                return await chat_completion_response_artifact(query=query, gen_queue=gen_queue, model_name=model_name_to_use, request=request,)
    except HTTPException as e:
        logger.error(e)
        return e


@router.post("/v1/chat/generate")
@router.post("/v1/completions")
async def generate(request: Request, query: GenerateQuery):
    global default_model_name
    if query.model and llm_dict.get(query.model):
        model_name_to_use = query.model
    else:
        model_name_to_use = default_model_name

    llm = llm_dict[model_name_to_use]["model"]

    gen_queue = GenQueue()  # this queue will hold the result for this generation

    # start the generation task
    asyncio.create_task(llm.chat_raw(
        prompt=query.prompt,
        max_tokens=query.max_tokens,
        gen_queue=gen_queue,
        request=request,
    ))

    if query.stream:
        # EventSourceResponse take iterator so need to handle at here
        return EventSourceResponse(completion_response_stream(request, gen_queue, model_name=model_name_to_use, request=request,))
    else:
        return await completion_response(gen_queue, model_name=model_name_to_use, request=request,)


@router.post("/v1/embeddings")
async def embeddings(request: Request, query: EmbeddingRequest):
    global default_model_name
    embedding_model = llm_dict[default_model_name]["model"]

    # for embedding, hard enforcement of matching model name
    if query.model != embedding_model.model_name:
        raise HTTPException(status_code=400,
                            detail=f"Embedding model {query.model} is not found. Current loaded model is {embedding_model.model_name}")

    return await embedding_model.text_embeddings(
        query=query,
    )


@router.get("/v1/models")
async def get_models(request: Request):
    data = []
    for model_name in llm_dict.keys():
        data.append(ModelObject(id=model_name))

    return ModelObjectResponse(data=data)


@router.options("/v1/chat/completions")
async def options_handler(request: Request):
    return JSONResponse(
        status_code=200,
        content={"message": "Local OpenAI response for OPTIONS request"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "",
        },
    )

@router.post("/v1/audio/transcriptions", response_model=TranscriptionResponse)
async def create_transcription(
    file: UploadFile = File(..., description="The audio file to transcribe."),
    model: str = Form(..., description="ID of the model to use."),
    language: Optional[str] = Form(None, description="The language of the input audio in ISO-639-1 format."),
    prompt: Optional[str] = Form(None, description="An optional text to guide the model's style or continue a previous segment."),
    response_format: Optional[Literal["json", "text", "srt", "verbose_json", "vtt"]] = Form("json", description="The format of the output."),
    temperature: Optional[float] = Form(0.0, description="The sampling temperature, between 0 and 1."),
    timestamp_granularities: Optional[List[Literal["word", "segment"]]] = Form(["segment"], description="The timestamp granularities for transcription.")
):
    """
    Transcribe an audio file into the input language.
    """

    global stt_dict

    stt_to_use = stt_dict.get(model).get("model")

    if stt_to_use is None:
        raise HTTPException(status_code=400, detail="Model not found")

    if response_format not in  ["json", "verbose_json"]:
        raise HTTPException(status_code=400, detail="Invalid response format, currently only json format supported")

    include_segments = False
    if response_format == "verbose_json" and "segment" in timestamp_granularities:
        include_segments = True

    transcribed_object = await stt_to_use.transcribe_async(
        audio=file.file,
        init_prompt=prompt,
        temperature=temperature,
        language=language,
        include_segments=include_segments,
    )

    return transcribed_object

from concurrent.futures import ThreadPoolExecutor

@router.post("/v1/audio/speech")
async def create_speech(request: TTSRequest):
    # Validate the model, voice, and input length
    # if request.model not in SUPPORTED_MODELS:
    #     raise HTTPException(status_code=400, detail="Unsupported model")
    # if request.voice not in SUPPORTED_VOICES:
    #     raise HTTPException(status_code=400, detail="Unsupported voice")
    if len(request.input) > 4096:
        raise HTTPException(status_code=400, detail="Input text exceeds the maximum length of 4096 characters")

    global tts_dict
    tts_to_use = tts_dict.get(request.model).get("model")

    if len(request.input) > 4096:
        raise HTTPException(status_code=400, detail="Input text exceeds 4096 characters")

    if request.speed < 0.25 or request.speed > 4.0:
        raise HTTPException(status_code=400, detail="Speed must be between 0.25 and 4.0")

    # Get the current event loop at the start
    loop = asyncio.get_running_loop()

    # Generate audio asynchronously
    #voice_desc = get_voice_description(request.voice)
    sampling_rate, audio_data = await tts_to_use.text_to_speech(
        text = request.input,
        stream = False,
        batching = True,
        batch_size = 3
    )

    # Convert to desired format using a thread pool
    buffer = io.BytesIO()
    if request.response_format not in ["mp3", "wav", "flac"]:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {request.response_format}")

    # Write audio data to buffer in a thread pool
    await loop.run_in_executor(
        None,
        lambda: sf.write(buffer, audio_data, sampling_rate, format=request.response_format)
    )
    buffer.seek(0)

    # Set the appropriate media type
    media_types = {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "flac": "audio/flac"
    }

    return StreamingResponse(
        buffer,
        media_type=media_types[request.response_format],
        headers={
            "Content-Disposition": f'attachment; filename="speech.{request.response_format}"'
        }
    )



@router.post("/load_model")
def load_model(model_spec: ModelParser):
    """
    model_spec is model specification coming from cli
    it might not have all the properties required for the model to be loaded
    the config_manager below contain all the models properties
    """
    global config_manager, llm_dict, stt_dict, tts_dict

    # get the config from the yml
    model_name = model_spec.model_id
    model_config = config_manager.get_model_config(model_name)

    if not model_config:
        raise Exception(f"Model config for '{model_name}' not exist in ~/gallama/model_config.yaml")

    # load the model with config from the model_spec and yml. model_spec comes from cli
    if model_config["backend"] in ["exllama", "llama_cpp", "transformers"]:  # llm loading
        prompt_eng = PromptEngine(prompt_format=model_config["prompt_template"])

        if model_spec.draft_model_id:
            draft_model_config = config_manager.get_model_config(model_spec.draft_model_name)
            if not draft_model_config:
                raise Exception(
                    f"Model config for '{model_spec.draft_model_name}' not exist in ~/gallama/model_config.yaml")
        else:
            draft_model_config = {}


        # load LLM model
        model_class_dict = {
            "exllama": ModelExllama,
            "llama_cpp": ModelLlamaCpp,
            "transformers": ModelTransformers,
        }

        model_class_to_use = model_class_dict[model_config["backend"]]

        llm = model_class_to_use(
            model_spec=model_spec,
            model_config=model_config,
            draft_model_config=draft_model_config,
            eos_token_list_from_prompt_template=prompt_eng.eos_token_list,
        )

        # update dict
        llm_dict[model_name] = {
            "model": llm,
            "prompt_engine": prompt_eng,
        }
    elif model_config["backend"] == "embedding":   # embedding model
        from gallama.backend.embedding.embedding import EmbeddingModel

        llm = EmbeddingModel(
            model_id=model_config["model_id"],
            model_name=model_name,
            model_spec=model_spec,
            model_config=model_config,
        )

        # update dict
        llm_dict[model_name] = {
            "model": llm,
        }
    elif model_config["backend"] == "faster_whisper":   # embedding model
        from gallama.backend.stt import ASRProcessor, ASRFasterWhisper

        stt_base = ASRFasterWhisper(
            modelsize="large-v3-turbo",  # TODO hardcode for testing
            cache_dir=None,
            model_dir=None,
            quant= "16.0",
            # model_id=model_config["model_id"],
            # model_name=model_name,
            # model_spec=model_spec,
            # model_config=model_config,
        )

        stt = ASRProcessor(asr=stt_base)

        # update dict
        stt_dict[model_name] = {
            "model": stt,
        }
    elif model_config["backend"] == "gpt_sovits":   # embedding model
        from gallama.backend.tts import TTS_GPT_SoVITS
        logger.info("load TTS")
        tts = TTS_GPT_SoVITS(
            model_spec=model_spec,
            model_config=model_config,
        )

        # update dict
        tts_dict[model_name] = {
            "model": tts,
        }

    logger.info("Loaded: " + model_name)




@router.post("/delete_model")
def delete_model(model_name: str):

    if model_name in llm_dict.keys():
        del llm_dict[model_name]

        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Run garbage collection
        gc.collect()

        logger.info("Deleted model: " + model_name)


@router.get("/status")
async def get_status():
    logger.info("Status endpoint called")
    return {"status": "ready" if model_ready else "loading"}


# Add a health check endpoint
@router.get("/health")
async def health_check():
    return {"status": "healthy"}


async def startup_event():
    # run some dummy generation so that cache is initialized
    logger.info("Generator initialization")
    global llm_dict
    gen_queue = GenQueue()

    for _model_name, _model in llm_dict.items():
        if _model.get("prompt_engine"):     # this is an LLM not embedding model
            llm = _model["model"]
            await llm.chat_raw(
                prompt=f"{ARTIFACT_SYSTEM_PROMPT}\nWrite a 500 words story on Llama",
                # stream=False,
                max_tokens=200,
                gen_queue=gen_queue,
                quiet=True,
                request=None,
            )

    gen_queue = None
    logger.info("Generator warmed up")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup code
    await startup_event()
    yield


def make_server(args):
    global logger
    global model_ready
    global llm_dict, stt_dict, tts_dict
    global default_model_name
    global draft_spec_dict
    # load yaml file of model info
    logger.info(args)
    model_dict = {}
    draft_spec_dict = {}    # for speculative decoding

    # parse cli argument to get model info
    if args.model_id:
        for item in args.model_id:
            model_dict.update(item)
        model_spec = ModelParser.from_dict(model_dict)

        # set the env for log to zmq
        # Note: This assumes you want to use the first model's ID if multiple are provided
        os.environ['MODEL_NAME'] = model_spec.model_id
        os.environ['MODEL_PORT'] = str(args.port)

    # set logger level
    if args.verbose:
        logger.setLevel(DEBUG)
        os.environ["LOCAL_OPEN_AI_VERBOSE"] = '2'   # turn on verbosity for all

    logger.info("Parsed Arguments:" + str(args))  # Debug statement

    # initialize fastapi

    # set ready status to False
    model_ready = False

    if args.detached:
        # send logging to zmq so that it will show in the parent log
        logger = get_logger(name="child", to_zmq=True, to_console=False)
    else:
        # keep the default logger declared on top
        pass

    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'per_thread_default_stream'

    app = FastAPI(lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], #origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    @app.middleware("http")
    @app.middleware("https")
    async def log_requests(request: Request, call_next):
        try:
            # Log request details for specific methods
            if request.method in ("POST", "PUT", "PATCH"):  # Methods that typically have a body
                # Use the helper function to parse the request body
                request_content = await parse_request_body(request, return_full_body=True)

                if isinstance(request_content, str):
                    try:
                        # Attempt to format as JSON if possible
                        parsed_content = json.loads(request_content)
                        request_content = json.dumps(parsed_content, indent=2)
                    except json.JSONDecodeError:
                        # Leave as raw string if it's not JSON
                        pass

                logger.debug(
                    f"API Request:\n"
                    f"Method: {request.method}\n"
                    f"URL: {request.url}\n"
                    f"Headers: {dict(request.headers)}\n"
                    f"Body: {request_content if request_content else 'No Body'}"
                )

            # Proceed with the request
            response = await call_next(request)

        except RequestValidationError as e:
            logger.debug(f"Validation error:\n{e}")
            response = JSONResponse(status_code=422, content={"detail": "Validation error"})

        return response

    llm_dict = {}
    stt_dict = {}
    tts_dict = {}

    # make relevant parameter global for endpoint function to use
    # load LLM model
    if args.model_id:
        model_dict = {}
        for item in args.model_id:
            model_dict.update(item)
        model_spec = ModelParser.from_dict(model_dict)
        default_model_name = model_spec.model_id

        # load model
        load_model(model_spec)

    model_ready = True
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)    # disable log_config to use our custom logger


def parse_dict(arg):
    """Parses a key=value string and returns a dictionary."""
    result = {}
    for pair in arg.split():
        key, value = pair.split('=')
        result[key] = value.strip("'")  # Strip single quotes here as well
    return result


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Launch local OpenAI model")
    arg_parser.add_argument("-id", "--model_id", nargs='+', type=parse_dict, default=None)
    arg_parser.add_argument('-v', "--verbose", action='store_true', help="Turn on more verbose logging")
    arg_parser.add_argument('-d', "--detached", action='store_true', help="Log to ZeroMQ")
    arg_parser.add_argument("--host", type=str, default="127.0.0.1", help="The host to bind to.")
    arg_parser.add_argument('-p', "--port", type=int, default=8000, help="The port to bind to.")
    arg_parser.add_argument("--reload", action="store_true", help="Enable auto-reload.")

    args = arg_parser.parse_args()

    make_server(args)
