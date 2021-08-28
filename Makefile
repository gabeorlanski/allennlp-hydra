# This was copied from AllenNLP to allow use of the typechecking.
.PHONY : typecheck
typecheck :
	mypy allennlp tests scripts --cache-dir=/dev/null