export function AI_VIZ_LOG(msg,data){
    const log_txt_area = document.getElementById('TXT_AI_VIZ_LOG_1')
    // debugger;
    log_txt_area.value += '\n'+ msg;
    console.log(msg,data)
    //PURPOSE: TOOL IN AI to output META-STATE
    // console.log(msg)
}
