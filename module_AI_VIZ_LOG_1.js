//PURPOSE: TOOL IN AI to output META-STATE to UI (and console).
export function AI_VIZ_LOG(msg,data1,data2){
    data1 = (!data1)? '' : data1; //trap undefined
    data2 = (!data2)? '' : data2; //trap undefined
    const log_txt_area = document.getElementById('TXT_AI_VIZ_LOG_1')
    log_txt_area.value += '\n'+ msg;
    console.log(msg,data1,data2)
}
