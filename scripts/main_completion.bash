# Bash completion for the Image Processor main controller.

_IMAGE_PROCESSOR_ROOT="/home/vahram/Desktop/image_Processor"
_IMAGE_PROCESSOR_MAIN="${_IMAGE_PROCESSOR_ROOT}/scripts/main.py"
_IMAGE_PROCESSOR_PYTHON="${_IMAGE_PROCESSOR_ROOT}/.venv/bin/python"

_image_processor_commands()
{
    "${_IMAGE_PROCESSOR_PYTHON}" "${_IMAGE_PROCESSOR_MAIN}" completion --words
}

_image_processor_complete_command()
{
    local current="${COMP_WORDS[COMP_CWORD]}"
    local commands
    commands="$(_image_processor_commands)"
    COMPREPLY=($(compgen -W "${commands}" -- "${current}"))
}

_image_processor_main_index()
{
    local index
    local candidate

    for ((index = 1; index < COMP_CWORD; index++)); do
        candidate="${COMP_WORDS[index]}"

        case "${candidate}" in
            "${_IMAGE_PROCESSOR_MAIN}"|\
            "${_IMAGE_PROCESSOR_ROOT}/main.py"|\
            scripts/main.py|./scripts/main.py|main.py|./main.py)
                printf '%s\n' "${index}"
                return 0
                ;;
        esac
    done

    return 1
}

_image_processor_python_completion()
{
    local main_index

    if ! main_index="$(_image_processor_main_index)"; then
        COMPREPLY=()
        return 0
    fi

    if ((COMP_CWORD == main_index + 1)); then
        _image_processor_complete_command
    else
        COMPREPLY=()
    fi
}

ocr()
{
    "${_IMAGE_PROCESSOR_PYTHON}" "${_IMAGE_PROCESSOR_MAIN}" "$@"
}

complete -o bashdefault -o default -F _image_processor_complete_command ocr
complete -o bashdefault -o default -F _image_processor_python_completion \
    python python3 \
    "${_IMAGE_PROCESSOR_PYTHON}" \
    .venv/bin/python
