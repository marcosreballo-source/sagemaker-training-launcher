"""Sessão AWS.

Credenciais nunca são lidas do código: a sessão usa a cadeia padrão do boto3
(variáveis de ambiente, ~/.aws/credentials, perfil nomeado, SSO ou role da instância).
"""

from __future__ import annotations

import boto3

from smlaunch.config import AwsConfig


def make_session(aws: AwsConfig) -> boto3.Session:
    return boto3.Session(profile_name=aws.profile, region_name=aws.region)
