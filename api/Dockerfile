FROM python:3.13

RUN useradd -m -u 1000 user

WORKDIR /app

RUN mkdir -p logs

RUN pip install uv

# RUN uv pip install --system git+https://github.com/mpilhlt/llamore.git[api]
RUN uv pip install --system "git+https://github.com/mpilhlt/llamore.git@feat/add_webservice_api#egg=llamore[api]"

COPY --chown=user api.py .

EXPOSE 7860

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860"]

