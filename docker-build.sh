tag="$(git rev-parse --abbrev-ref HEAD | sed "s/.*\///")"
gitsha="$(git rev-parse --short HEAD)"

echo $tag
echo $gitsha

docker build . -t WITHHELD/mp:$tag --build-arg COMMIT=$gitsha
docker push WITHHELD/mp:$tag
