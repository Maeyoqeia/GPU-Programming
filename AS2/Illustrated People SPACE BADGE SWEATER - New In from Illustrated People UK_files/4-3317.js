/**
* FLXone data collection generated at 2015-07-08 06:23:58
*/
var flx1=function(f){function l(a){var c=document,b=c.createElement("script");b.async=!0;b.defer=!0;b.src=a;c.getElementsByTagName("head")[0].appendChild(b)}function n(a){var c={};if(!a)a:{var b=document.getElementsByTagName("script"),e="/"+f.m+"-"+f.id+".js";a=0;for(var d=b.length;a<d;a++)if(b[a].src&&-1!==b[a].src.indexOf(e)){a=b[a].src;break a}a=""}if(a&&-1!==a.indexOf("?"))for(b=RegExp("[+]","g"),e=RegExp("([^&=]+)=?([^&]*)","g"),d=a.split("?")[1];a=e.exec(d);)c[decodeURIComponent(a[1].replace(b," "))]=decodeURIComponent(a[2].replace(b," "));return c}var g=[],m=null;return{pxl:function(a){a="//"+flx1.cnf.d+a;var c="",b=n(),e={m:1,id:1,d:1},d;for(d in b)b.hasOwnProperty(d)&&(e[d]||(c+="&"+d+"="+encodeURIComponent(b[d])));d=a+c+"&r="+encodeURIComponent(document.referrer)+"&eurl="+encodeURIComponent(document.location.href)+"&rndm="+1E16*Math.random();l(d)},_pxl:l,cnf:f,log:function(a){window.console&&console.log&&console.log(a)},data:function(a,c,b,e){g.push({k:a,v:c,custom:"undefined"===typeof b||null===b?!1:b});null===m&&(m=setTimeout(function(){m=null;if(null!==g&&0<g.length){var a="/px?id="+flx1.cnf.id+"&m="+flx1.cnf.m,b=a,c={},h="",f;for(f in g)if(g.hasOwnProperty(f))try{var k=g[f];!0===k.custom?(c[k.k]=k.v,h="&data="+encodeURIComponent(JSON.stringify(c))):b+="&"+k.k+"="+encodeURIComponent(k.v);2048<=b.length+h.length&&(flx1.pxl(b+h),b=a)}catch(l){flx1.log(l)}b===a&&""===h||flx1.pxl(b+h);g=[]}"function"===typeof e&&e()},100))},getUrlParams:n}}({id:"3317",m:"4",d:"go.flx1.com"});flx1.pxl("/px?id="+flx1.cnf.id+"&m="+flx1.cnf.m);/**
* Fire interaction events after a couple of seconds on this page
*/
(function() {
    try {
        var ivs = [5,10,20,30,60,90,120,180,240,300];
        for (var k in ivs) {
            if (ivs.hasOwnProperty(k)) {
                (function(iv) {
                    setTimeout(function() {
                        flx1.pxl("/ia?id="+flx1.cnf.id+"&m="+flx1.cnf.m+"&it=4&iv="+iv);
                    },iv*1000);
                })(ivs[k]);
            }
        }
    } catch (e) {flx1.log(e);}
})();