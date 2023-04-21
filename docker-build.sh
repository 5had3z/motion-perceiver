tag="$(git rev-parse --abbrev-ref HEAD | sed "s/.*\///")"
gitsha="$(git rev-parse --short HEAD)"

echo $tag
echo $gitsha

docker build . -t $REGISTRY_URL/mp:$tag --build-arg COMMIT=$gitsha
docker push $REGISTRY_URL/mp:$tag
