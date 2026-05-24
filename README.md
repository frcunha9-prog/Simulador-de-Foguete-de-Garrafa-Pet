# 🚀 Simulador de Foguete PET

Web app que calcula o **ângulo de lançamento ótimo** para máximo alcance
horizontal de um foguete de garrafa PET propulsionado a água sob pressão.

## Executar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

Abre em `http://localhost:8501` no navegador.

## Publicar de graça (recomendado)

### Streamlit Community Cloud

1. Crie um repositório público no GitHub e suba os 3 arquivos:
   - `app.py`
   - `requirements.txt`
   - `README.md`

2. Vá em https://streamlit.io/cloud e faça login com sua conta GitHub.

3. Clique em **"New app"**, selecione o repositório, branch `main`,
   arquivo principal `app.py`, e clique em **Deploy**.

4. Em ~2 minutos o app fica no ar com URL do tipo
   `https://seunome-foguete-pet.streamlit.app`.

5. No celular, abra essa URL no Chrome, depois **Menu → Adicionar à tela
   inicial**. Vira ícone como se fosse app nativo.

### Hugging Face Spaces (alternativa)

1. Crie um Space novo em https://huggingface.co/new-space.
2. Escolha SDK **Streamlit**.
3. Suba os mesmos 3 arquivos.
4. O app fica em `https://huggingface.co/spaces/seunome/foguete-pet`.

## Estrutura

- `app.py` — interface Streamlit + núcleo de simulação (autocontido).
- `requirements.txt` — dependências (numpy, scipy, matplotlib, streamlit).

## Modelo físico

Três fases:

1. **Empuxo por água** — expansão adiabática do ar + Bernoulli.
2. **Empuxo por ar** — escoamento isentrópico compressível com choke.
3. **Voo balístico** — gravidade + arrasto aerodinâmico em 2D.

O tubo de lançamento é modelado explicitamente: enquanto dentro do tubo,
o movimento é confinado 1D na direção do lançamento.

## Calibração

O parâmetro mais incerto é o `Cd` do corpo. Para resultados confiáveis:

1. Faça 3 lançamentos verticais reais.
2. Meça o apogeu com câmera/tracker.
3. Ajuste o `Cd` no app até o apogeu previsto bater com o observado (±10%).
