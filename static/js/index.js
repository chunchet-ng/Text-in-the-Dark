function toggle_btn(btn) {
  if(btn.classList.contains("btn-primary")){
    btn.classList.add("btn-success")
    btn.classList.remove("btn-primary");
  }
  else{
    btn.classList.add("btn-primary")
    btn.classList.remove("btn-success");
  }
}

function toggle_div(div) {
  if(div.classList.contains("border-primary")){
    div.classList.add("border-success")
    div.classList.remove("border-primary");
    div.classList.add("text-success")
    div.classList.remove("text-primary");
  }
  else{
    div.classList.add("border-primary")
    div.classList.remove("border-success");
    div.classList.add("text-primary")
    div.classList.remove("text-success");
  }
}

function copy_text(name) {

  if (name == 'old') {
    var copyText = document.getElementById("bibtex_old");
    var btn = document.getElementById("bibtex_btn_old");
    var icon_div = document.getElementById("icon_div_old");
    var bibtex_div = document.getElementById("bibtex_div_old");
  }
  else {
    var copyText = document.getElementById("bibtex_new");
    var btn = document.getElementById("bibtex_btn_new");
    var icon_div = document.getElementById("icon_div_new");
    var bibtex_div = document.getElementById("bibtex_div_new");
  }

  const check_icon = document.createElement('i');
  check_icon.classList.add('fa-solid');
  check_icon.classList.add('fa-check');

  const copy_icon = document.createElement('i');
  copy_icon.classList.add('fa-solid');
  copy_icon.classList.add('fa-copy');

  toggle_btn(btn)
  toggle_div(bibtex_div)
  icon_div.innerHTML = null
  icon_div.appendChild(check_icon)

  navigator.clipboard.writeText(copyText.innerText).then(
    setTimeout(function() {
      toggle_btn(btn)
      icon_div.innerHTML = null
      icon_div.appendChild(copy_icon)
      toggle_div(bibtex_div)
    }, 2000)
  );

}