
from aicar.utils.voice import pyrec, wav2pcm, baidu_ai


def voice():
    pyrec.rec("1.wav")  # 录音并生成wav文件,使用方式传入文件名

    pcm_file = wav2pcm.wav_to_pcm("1.wav")  # 将wav文件 转换成pcm文件 返回 pcm的文件名

    res_str = baidu_ai.audio_to_text(pcm_file)  # 将转换后的pcm音频文件识别成 文字 res_str
    return res_str
