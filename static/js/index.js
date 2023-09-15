function copy_text() {
  var copyText = document.getElementById("bibtex");
  navigator.clipboard.writeText(copyText.innerText);
}