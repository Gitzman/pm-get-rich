FROM nginx:alpine
COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY index.html /usr/share/nginx/html/
COPY comparison.html /usr/share/nginx/html/
COPY generalization.html /usr/share/nginx/html/
COPY eval.html /usr/share/nginx/html/
COPY backtest.html /usr/share/nginx/html/
COPY data/ /usr/share/nginx/html/data/
EXPOSE 8080
