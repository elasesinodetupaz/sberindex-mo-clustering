# Сертификаты Минцифры для скачивания справочника МО

Эти файлы нужны только скрипту `src/24b_geometry.py`, и только если справочник границ МО
(`t_dict_municipal_districts_poly.gpkg`) не найден локально и архив скачивается с
`https://www.sberbank.com/common/files/t_dict_municipal.rar`. Сервер подписан цепочкой
Russian Trusted Root CA и Russian Trusted Sub CA 2024; этих корней нет в системном хранилище macOS.

## Файлы

| файл | что это |
|---|---|
| `russian_trusted_root_ca.pem` | корневой сертификат Russian Trusted Root CA |
| `russian_trusted_sub_ca_2024.pem` | промежуточный сертификат Russian Trusted Sub CA 2024 |
| `ca_bundle.pem` | оба сертификата подряд (корень, затем Sub CA 2024), каждый PEM с переводами строк |

## Откуда взяты

Файлы получены 04.10.2026 с официального домена `gu-st.ru` по ссылкам со страницы
`https://www.gosuslugi.ru/landing/crt` (соединение с проверкой TLS):

- `https://gu-st.ru/content/downloads/Russian_Trusted_Root_CA.cer`
- `https://gu-st.ru/content/downloads/Russian_Trusted_Sub_CA_2024.cer`

Исходные файлы уже были в формате PEM; в репозиторий они записаны заново из DER (перевод строк LF).

## Отпечатки sha256 (по DER)

- Корень: `D2:6D:2D:02:31:B7:C3:9F:92:CC:73:85:12:BA:54:10:35:19:E4:40:5D:68:B5:BD:70:3E:97:88:CA:8E:CF:31`
- Sub CA 2024: `21:55:78:50:36:C9:00:DB:B5:F1:BB:2A:15:69:C8:0C:55:59:5B:D6:BF:94:86:7A:29:BB:DD:BC:7D:88:A3:F2`

Отпечаток корня совпал побайтно с опубликованным на странице
`https://developers.sber.ru/docs/en/smarthome/c2c/ca` (страница обновлена 23.12.2022; у корня указан
SHA-256 `D2 6D 2D 02 ... CA 8E CF 31`). Отпечаток Sub CA 2024 на официальной странице Минцифры не
опубликован, поэтому он проверен только через подпись корня: `openssl verify -CAfile
certs/russian_trusted_root_ca.pem certs/russian_trusted_sub_ca_2024.pem` дал `OK`, а серверный
сертификат sberbank.com проходит проверку с корнем как единственным якорем и Sub CA 2024 как
промежуточным.

## Как используются

- Бандл подставляется только в запрос скачивания: `ssl.create_default_context(cafile="certs/ca_bundle.pem")`.
  Такой контекст не загружает системное хранилище.
- Сертификаты не добавляются в системное хранилище и в хранилище браузера.
- `curl` от Apple (SecureTransport) не поддерживает `--cacert`, поэтому скачивание выполняет модуль `ssl`
  из Python.
- Срок действия: корень до 27.02.2032, Sub CA 2024 до 19.07.2029; после этого файлы нужно получить заново
  с официальной страницы и сверить отпечатки.
