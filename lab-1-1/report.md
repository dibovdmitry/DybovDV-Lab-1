# МИНИCTEPCTBO НАУКИ И ВЫСШЕГО ОБРАЗОВАНИЯ  
РОССИЙСКОЙ ФЕДЕРАЦИИ 
Федеральное государственное автономное  
образовательное учреждение высшего образования 
«Северо-Кавказский федеральный университет» 
 
Департамент цифровых, робототехнических систем и электроники института перспективной инженерии

Отчет по лабораторной работе 1
Подготовка рабочего окружения для MLOps и основы трекинга экспериментов с использованием MLflow
Дата: 2025-11-14 
Семестр: [2 курс 1 полугодие - 3 семестр] 
Группа: ПИН-м-о-24-1 
Дисциплина: Технологии программирования 
Студент: Дыбов Д.В.

Цель работы
Освоение базовых принципов установки и настройки рабочего окружения для Data Science и MLOps на базе Anaconda (conda) и Docker; получение практических навыков управления виртуальными окружениями Python, работы с conda.

Теоретическая часть
Краткие изученные концепции:

Виртуальные окружения Python (Miniconda/conda): зачем нужны, создание и активация окружения, установка зависимостей.

Docker: образы, контейнеры, репозитории, работа без sudo через группу docker.

JupyterLab в контейнере: запуск и взаимодействие через браузер.

MLflow Tracking: запуск сервера, создание эксперимента, логирование параметров, метрик и артефактов (графиков) и просмотр результатов в веб-интерфейсе.

Практическая часть
Выполненные задачи
- [x] Задача 1: Установить Ubuntu 24.04.03 LTS в VirtualBox и подготовить среду для Docker.

- [x] Задача 2: Установить Miniconda, создать и активировать виртуальное окружение mlops-lab.

- [x] Задача 3: Установить пакеты pandas, scikit-learn, matplotlib, jupyterlab и проверить импорт.

- [x] Задача 4: Установить Docker Engine: добавить GPG-ключ, настроить репозиторий, установить engine.

- [x] Задача 5: Запустить тестовый Docker-образ и обучиться базовым командам: ps, ps -a, images, rm.

- [x] Задача 6: Запустить JupyterLab в Docker-контейнере и проверить доступ по http://127.0.0.1:9999.

#Ключевые фрагменты кода

# создание и активация окружения conda
conda create -n mlops-lab python=3.10 -y
conda activate mlops-lab

# установка зависимостей
conda install pandas scikit-learn matplotlib jupyterlab -y
pip install mlflow

# базовые docker-команды
sudo apt-get update
# добавить GPG ключ и репозиторий Docker (прим. команды)
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
# установка Docker Engine
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io -y
# добавить пользователя в группу docker
sudo usermod -aG docker $USER

# запуск JupyterLab в контейнере (пример)
docker run -p 9999:8888 jupyter/base-notebook



python
import mlflow
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

mlflow.set_experiment("mlops_lab_experiment")
with mlflow.start_run():
    X, y = make_regression(n_samples=100, n_features=1, noise=10.0, random_state=42)
    model = LinearRegression().fit(X, y)
    preds = model.predict(X)
    mse = mean_squared_error(y, preds)
    mlflow.log_metric("mse", mse)
    mlflow.log_param("noise", 10.0)
    plt.scatter(X, y)
    plt.plot(X, preds, color="red")
    plt.savefig("prediction_plot.png")
    mlflow.log_artifact("prediction_plot.png")
Результаты выполнения
Пример работы программы / выводы команд
Создано и активировано окружение mlops-lab.

Установлены пакеты pandas, scikit-learn, matplotlib, jupyterlab — импорт pandas прошёл успешно.

Docker установлен и проверен запуском тестового образа (контейнер вывел "Hello from Docker!" / успешное сообщение запуска).

Пользователь добавлен в группу docker — запуск docker без sudo подтверждён.

Запущен контейнер ubuntu:24.04 в интерактивном режиме, выполнены команды ls / и cat /etc/os-release — информация об ОС подтверждена.

Просмотр контейнеров: docker ps и docker ps -a; просмотр образов: docker images; удаление контейнера: docker rm <id> — отработало корректно.

JupyterLab запущен в контейнере, доступен по адресу http://127.0.0.1:9999; в ноутбуке выполнён тестовый код.

MLflow установлен; Tracking Server запущен и доступен в браузере (порт по умолчанию/указанный при запуске).

Создан файл mlflow_basic.py, запущен несколько раз с изменением параметров; в MLflow UI появился новый эксперимент, отображены запуски, логированные метрики и артефакты (включая построенный график).

Тестирование
[x] Модульные тесты — не применялись (задача лабораторная, проверка окружения).

[x] Интеграционные тесты — проверены интеграции: conda environment ↔ Docker контейнеры ↔ JupyterLab ↔ MLflow Tracking.

[x] Производительность — не в фокусе данной работы; операции по установке и запуску прошли в пределах ожидаемого времени.

Приложения
Ключевые команды установки и проверки (см. блок "Ключевые фрагменты кода").

Отчёт о результатах в MLflow UI: созданный эксперимент, список запусков, подробности запуска (params/metrics), вкладка Artifacts с сохранённым графиком.

Скрипт mlflow_basic.py (см. фрагмент выше); при необходимости прикладывается полный файл в папке src/.

Выводы
Рабочее окружение для MLOps было успешно подготовлено: настроены conda-окружение и Docker.

JupyterLab и MLflow корректно запущены внутри/взаимодействуют с Docker-контейнерами; эксперименты логируются и визуализируются в MLflow UI.

Получены практические навыки управления окружениями, установки зависимостей, базовой работы с Docker и трекинга экспериментов (логирование параметров, метрик и артефактов).
