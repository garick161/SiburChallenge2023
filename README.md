Ссылки на ноутбуки
https://nbviewer.org/github/garick161/SiburChallenge2023/blob/master/notebooks/research_review_part1.ipynb
https://nbviewer.org/github/garick161/SiburChallenge2023/blob/master/notebooks/research_review_part2.ipynb

![Group 8 (1)](https://github.com/garick161/SiburChallenge2023/assets/114688542/5a26d18d-36a7-4787-b7ac-3daffc9a90aa)

Данный кейс представлен нефтехимической компанией Сибур на чемпионате Sibur Challenge 2023 по направлению "Видеоаналитика".\
Более подробно вы можете ознакомится с материалами на платформе AI Today. Вся информация находится свободном доступе.

https://platform.aitoday.ru/event/9\

__*Цитата из условия задачи:*__

>Необходимо создать модель распознавания действий с вагон-цистерной на эстакаде, которая пошагово отслеживает жизненный цикл вагон-цистерны (въезд на эстакаду, подготовка к сливу, слив, подготовка к отъезду, выезд поезда и т. д.).

>Для решения задачи вам предоставлен набор видеороликов, каждый из которых относится к одному из четырех классов:

>`bridge_down` - класс "мостик опущен",

>`bridge_up` - класс "мостик поднят",

>`no_action` - класс "нет операций в кадре",

>`train_in_out` - класс "въезд/выезд цистерны".

>В одном ролике может быть ровно один класс. Особенность задачи заключается в том, что модель распознавания не должна требовать дорогой и долгой дотренировки или сбора дополнительных данных для развертывания системы видеоаналитики в новых локациях. Кроме того, модель должна быть устойчива к потере связи, т. е. пропущенным кадрам в видео.

>Метрика - F1 (macro).

Всего представлено 496 видеороликов дительностью не более 10 секунд с соотношением:\
`bridge_down` - 306\
`bridge_up` - 75\
`no_action` - 49\
`train_in_out` - 66

Формат видеороликов выглядит следующим образом
![readme_0_79](https://github.com/garick161/SiburChallenge2023/assets/114688542/b4020ee5-067e-426c-9ea8-b5f894c71459)

![readme_80_159](https://github.com/garick161/SiburChallenge2023/assets/114688542/3541e64c-0283-4e90-a793-6d02e2eb8f46)
