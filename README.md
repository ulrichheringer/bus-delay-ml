# bus-delay-ml

Prevê atraso de ônibus com características da rota, horário, clima, tráfego e carga de passageiros usando histórico sintético.

## Dados

O pipeline gera dados sintéticos determinísticos localmente, identificados como sintéticos em `pipeline.py`. Nenhum dado privado ou download externo é necessário. A divisão e a semente de avaliação são descritas junto às métricas abaixo.

## Instalação e execução

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python cli.py evaluate
python cli.py train --model model.joblib
python cli.py predict --model model.joblib
```

## Métricas obtidas

Seed 42: divisão cronológica do conjunto gerado — treino com 1.920 viagens e teste nas 480 viagens finais. MAE **2,3220 min**, RMSE **2,9843 min**, R² **0,6793**. Tráfego, chuva, carga e atrasos foram simulados; não são dados de transporte público real.

Limitação: dados e regras de geração simplificados; antes de uso real, substitua-os por dados com licença e proveniência adequadas e valide fora da amostra.
