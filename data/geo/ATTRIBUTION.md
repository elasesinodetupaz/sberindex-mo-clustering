# Источник и лицензия геометрии МО (data/geo/)

## Источник

Данные о границах и преобразованиях муниципальных образований. СберИндекс. Данные доступны по адресу https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities (данные скачаны 04.10.2026).

Data on the boundaries and transformations of municipalities. Sberindex. Available at https://sberindex.ru/ru/research/dataset-borders-and-changes-of-municipalities (data downloaded on 04.10.2026).

Набор построен СберИндексом по данным Росстата и OpenStreetMap (по странице набора). Условия лицензии OpenStreetMap (ODbL) в проекте не проверялись. Атрибуция: Contains data © OpenStreetMap contributors.

## Лицензия

Данные набора доступны по лицензии CC BY-SA 4.0 (по официальной странице набора): https://creativecommons.org/licenses/by-sa/4.0/legalcode

## Что изменено в файлах этой папки

Файлы `mo_national.geojson` и `mo_cities.geojson` являются адаптацией исходных полигонов и распространяются на тех же условиях (CC BY-SA 4.0). Изменения:

- выбраны полигоны, действующие в 2024-12 (`year_to = 9999`), всего 2594;
- геометрия упрощена одним вызовом `shapely.coverage_simplify` ко всему покрытию: допуск 0.005° для `mo_national.geojson` и 0.0005° для `mo_cities.geojson` (все действующие полигоны Москвы и Санкт-Петербурга, включая посёлки вне выборки);
- координаты округлены до 4 знаков (`mo_national.geojson`) и 5 знаков (`mo_cities.geojson`);
- атрибуты заменены: `territory_id`, `name`, `region_name` (из справочника), `in_sample` (входит ли МО в рабочую выборку проекта), в `mo_cities.geojson` ещё `city` (moscow или spb). Цвет и тип МО в файлы не записываются.

Параметры упрощения и метрики качества: `data/geo/geometry_report.md`. Скрипт: `src/24b_geometry.py`.

## Известные ошибки исходного справочника (не исправлены)

Числа по полному покрытию действующих полигонов:

- пар МО с перекрытием: 25 (в 13 обе стороны в рабочей выборке, в 12 участвует МО вне выборки);
- полигонов с неверными рёбрами по `shapely.coverage_is_valid` (`coverage_invalid_edges`): 35 из 2594; shapely.coverage_is_valid для всего покрытия: False;
- полигонов, недопустимых по `shapely.is_valid`: 0.

Перечень территориальных id: `data/geo/geometry_report.md`. Исходные ошибки не исправлялись.

## Что показали проверки упрощённых файлов (подробности в `data/geo/geometry_report.md`)

- `mo_national.geojson`: МО, недопустимых по `shapely.is_valid` после упрощения и округления: 10 (перечень и shapely.explain_validity в отчёте, не исправлялись); пар с перекрытием: 17;
- `mo_cities.geojson`: недопустимых МО 0; пар с перекрытием: 0;
- мелкие части МО, исчезнувшие при упрощении или округлении, перечислены в отчёте.

## Подпись для карты

Границы МО: СберИндекс, «Данные о границах и преобразованиях муниципальных образований» (скачано 04.10.2026), CC BY-SA 4.0; геометрия упрощена (shapely.coverage_simplify, допуск 0.005°, врезки Москвы и Санкт-Петербурга 0.0005°). Набор построен по данным Росстата и OpenStreetMap: Contains data © OpenStreetMap contributors.

