import functools
import os
import anthropic
import chainlit as cl
from chainlit.playground.providers import Anthropic

# Initialize the Anthropic client
anthropic_client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


@functools.cache
def construct_system_prompt():
    """Reads 'TWAC2023.txt', returns a system prompt string including the book's content."""
    with open('TWAC2023.txt', 'r') as file:
        book_content = repr(file.read())

    return ("You are an expert on Albert Wenger's The World After Capital. The "
            "full text of this book is available below. You will prioritize "
            "discussions on how the book's themes relate to the ethics of AI, "
            "reflecting Wenger's interest in this area. You will draw from the "
            "pages and chapters of the book to provide in-depth knowledge and "
            "engage in ethical considerations. You use the book's content as a "
            "foundation to explore the broader implications and challenges "
            "posed by AI, ensuring that conversations remain relevant and "
            "insightful with respect to current technological and societal "
            "trends. Your tone matches that used by the author in his blog "
            "(https://continuations.com) and twitter account "
            "(https://twitter.com/albertwenger). When quoting from the book, "
            "you will cite using the format: 'quote' (Ch. X[, 'Y']), where X "
            "is the chapter number and Y is the chapter name which is only "
            "included upon the first mention of a particular chapter. "
            "Crosscheck that your replies are accurate and directly supported "
            "by evidence from the book before responding.\n<book-content>\n"
            f"{book_content}\n</book-content>")


async def stream_claude_step(step, system_prompt, messages, initial_step=False):
    """Streams conversation steps with the Claude model."""
    settings = {"max_tokens": 1024, "model": "claude-3-5-sonnet-20241022"}

    async with step as current_step:
        async with anthropic_client.messages.stream(
            **settings,
            system=system_prompt,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                await current_step.stream_token(text)

    current_step.generation = cl.CompletionGeneration(
        formatted=messages,
        completion=current_step.output,
        settings=settings,
        provider=Anthropic.id,
    )

    response = [{"role": "assistant", "content": current_step.output}]

    if not initial_step:
        cl.user_session.set("prompt_history", messages + response)


@cl.on_chat_start
async def start_claude_conversation():
    """Initializes conversation with the Claude model."""
    cl.user_session.set("prompt_history", [])
    
    avatar_url = "https://www.anthropic.com/images/icons/apple-touch-icon.png"
    await cl.Avatar(name="Claude 3.5 Sonnet", url=avatar_url).send()

    step = cl.Step(name="Claude 3.5 Sonnet", type="llm", root=True)
    system_prompt = construct_system_prompt()
    messages = [{
        "role": "user",
        "content": ("Acknowledge that you understand the provided context with "
                    "the response, 'Hello, I am an expert on Albert Wenger's "
                    "book The World After Capital. What would you like to "
                    "know?'")
    }]

    await stream_claude_step(step, system_prompt, messages, initial_step=True)


@cl.on_message
async def on_message(message: cl.Message):
    """Responds to each message by calling Claude with the user's query."""
    await call_claude(message.content)


@cl.step(name="Claude", type="llm", root=True)
async def call_claude(query: str):
    """Calls the Claude model with the user's query and streams the response."""
    prompt_history = cl.user_session.get("prompt_history")

    step = cl.context.current_step
    system_prompt = construct_system_prompt()
    messages = prompt_history + [{"role": "user", "content": query}]

    await stream_claude_step(step, system_prompt, messages)
